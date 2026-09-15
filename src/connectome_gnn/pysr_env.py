"""Starting PySR, and saying so in results/ when it cannot start.

WHY THIS EXISTS. PySR's first import starts a Julia process, and on this cluster
that import has failed twice for reasons that have nothing to do with symbolic
regression -- both of which produced a figure whose equation column read
`[PySR unavailable]` while every other number in the directory looked normal:

  TMPDIR. The site sets TMPDIR=/scratch/$USER, which does not exist on the
  nodes. Julia's `tempdir()` throws inside `Pkg.Registry.update()` and the
  import aborts before any package loads.

  THE SYSTEM IMAGE, resolved against the CURRENT DIRECTORY. Launched from the
  repo, Julia looked for `<cwd>/../lib/julia/sys.so` -- `~/Graph/lib/julia/...`
  -- instead of the one beside its own executable, so PythonCall could not
  precompile. The same import from the home directory succeeded minutes
  earlier, which is what identified the cwd as the variable.

Neither is catchable in-process: a Julia failure of this kind is a signal 6
abort that takes the interpreter with it. So the import is PROBED IN A
SUBPROCESS first. One Julia start (about a minute) against a plot pass measured
in tens of minutes, and in exchange a run can never die inside a figure.

AND WHEN IT STILL FAILS, results/ SAYS SO, in exactly one file -- README.md --
carrying the reason, the environment it was tried in, and what to change. A
directory whose panels silently lack equations is indistinguishable from one
where the fits were run and found nothing.
"""

import os
import subprocess
import sys
import tempfile

_README = "README.md"

# module, reason, and the environment the successful (or failed) import used.
_STATE = {"checked": False, "module": None, "reason": None, "env": {}}


def _writable_dir(path):
    if not path:
        return False
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".connectome_gnn_probe")
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def prepare_environment(fallback=None):
    """Make TMPDIR real and the system image absolute. Returns what it changed.

    Both settings are process environment, so they must be in place BEFORE the
    first `import pysr` -- Julia reads them at startup and never again.
    """
    changed = {}
    tmp = os.environ.get("TMPDIR")
    if tmp and not os.path.isdir(tmp):
        if _writable_dir(tmp):
            changed["TMPDIR"] = f"created {tmp}"
        else:
            for cand in (fallback, tempfile.gettempdir(), "/tmp"):
                if _writable_dir(cand):
                    os.environ["TMPDIR"] = cand
                    changed["TMPDIR"] = f"{tmp} does not exist -> {cand}"
                    break
    return changed


def safe_cwd():
    """A directory Julia can resolve its own paths against.

    The home directory: the import that worked ran there, and unlike the repo it
    has no `../lib` for a relative system-image lookup to land in.
    """
    home = os.path.expanduser("~")
    return home if os.path.isdir(home) else tempfile.gettempdir()


def _probe(timeout=None):
    """Import pysr in a subprocess. Returns (ok, message).

    The timeout has to cover a COLD depot: the first import downloads Julia and
    precompiles SymbolicRegression and PythonCall, which took longer than 15
    minutes in the devcontainer and reported as a failure when the budget was
    900 s. Warm, the same import is seconds. Override with PYSR_PROBE_TIMEOUT
    when a site is slower still.
    """
    if timeout is None:
        try:
            timeout = float(os.environ.get("PYSR_PROBE_TIMEOUT", 2400))
        except ValueError:
            timeout = 2400
    code = ("import pysr, sys; "
            "sys.stdout.write('PYSR_VERSION=' + pysr.__version__)")
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=safe_cwd(),
                           capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return False, f"import pysr did not finish within {timeout}s"
    if r.returncode == 0 and "PYSR_VERSION=" in r.stdout:
        return True, r.stdout.split("PYSR_VERSION=")[-1].strip()
    tail = (r.stderr or r.stdout or "").strip().splitlines()
    reason = " | ".join(tail[-6:]) if tail else f"exit code {r.returncode}"
    if r.returncode < 0:
        reason = f"killed by signal {-r.returncode} -- {reason}"
    return False, reason


def on_lsf_outside_a_job():
    """True on a cluster host that is not inside a batch job -- i.e. a login node.

    Starting PySR there precompiles PythonCall, which pins a core at 100% CPU and
    hundreds of megabytes; the login-node monitor flags that at CPU >= 100% or
    RSS >= 1 GB and it earned a mail from the cluster admins on 2026-09-15. LSF
    sets LSB_JOBID inside a job and nowhere else, so its absence next to an LSF
    installation is the one reliable signal. The devcontainer has no LSF and is
    unaffected.
    """
    import shutil
    on_lsf = bool(os.environ.get("LSF_ENVDIR") or os.environ.get("LSF_SERVERDIR")
                  or shutil.which("bsub"))
    return on_lsf and not os.environ.get("LSB_JOBID")


def import_pysr(logger=None):
    """The PySR module, or None with the reason recorded. Probed once per run."""
    if _STATE["checked"]:
        return _STATE["module"]
    _STATE["checked"] = True
    if on_lsf_outside_a_job():
        _STATE["reason"] = ("refused to start PySR outside a batch job: this is "
                            "an LSF host with no LSB_JOBID, i.e. a login node. "
                            "Julia's precompilation is a full core and is "
                            "flagged there. Re-run under bsub.")
        _STATE["env"] = {"TMPDIR": os.environ.get("TMPDIR", "<unset>"),
                         "cwd_used": safe_cwd(), "python": sys.executable,
                         "adjusted": {}}
        if logger:
            logger.warning(_STATE["reason"])
        print(f"\033[91mPySR NOT started: {_STATE['reason']}\033[0m")
        return None
    changed = prepare_environment()
    _STATE["env"] = {"TMPDIR": os.environ.get("TMPDIR", "<unset>"),
                     "cwd_used": safe_cwd(), "python": sys.executable,
                     "adjusted": changed}
    ok, msg = _probe()
    if not ok:
        _STATE["reason"] = msg
        if logger:
            logger.warning(f"PySR unavailable: {msg}")
        print(f"\033[93mPySR unavailable: {msg}\033[0m")
        return None
    old = os.getcwd()
    try:
        os.chdir(safe_cwd())
        import pysr
        _STATE["module"] = pysr
        _STATE["env"]["version"] = getattr(pysr, "__version__", msg)
        # Said once, out loud: the equation columns further down are only
        # trustworthy if this line appeared, and its absence is exactly what a
        # reader of a finished log needs to notice.
        _ok = (f"PySR {_STATE['env']['version']} init passed "
               f"(TMPDIR={os.environ.get('TMPDIR', '<unset>')}, "
               f"imported from {safe_cwd()})")
        if logger:
            logger.info(_ok)
        print(f"\033[92m{_ok}\033[0m")
    except BaseException as exc:
        _STATE["reason"] = f"{type(exc).__name__}: {exc}"
        print(f"\033[91mPySR init FAILED after the probe passed: "
              f"{_STATE['reason']}\033[0m")
    finally:
        try:
            os.chdir(old)
        except OSError:
            pass
    return _STATE["module"]


def available():
    return _STATE["module"] is not None


def reason():
    return _STATE["reason"]


def write_status(log_dir):
    """results/README.md exists if and only if PySR could not start.

    Returns the path written, or None. Removing it on success is half the point:
    a stale README from a failed run would otherwise outlive the failure and
    describe a directory whose numbers did come from PySR.
    """
    results = os.path.join(str(log_dir), "results")
    path = os.path.join(results, _README)
    if not _STATE["checked"] or available():
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return None
    os.makedirs(results, exist_ok=True)
    env = _STATE["env"]
    text = f"""# PySR did not run for this run

The symbolic regression in `neuron<N>_panels.png` -- the `template` and `free`
rows beside each synapse -- is the only thing in this directory that needs PySR,
and it did not start. Every other number here (metrics.txt, the scatters,
parameter_error.png) comes from the closed-form template readout in
`connectome_gnn.metrics` and is unaffected.

## What failed

    {_STATE['reason']}

## The environment it was tried in

    TMPDIR    {env.get('TMPDIR')}
    cwd       {env.get('cwd_used')}
    python    {env.get('python')}
    adjusted  {env.get('adjusted') or 'nothing'}

## The two failures seen on this cluster, and their fixes

1. `ArgumentError: "/scratch/$USER" is not a directory` -- the site sets TMPDIR
   to a path that does not exist on the node. Julia's `tempdir()` throws inside
   `Pkg.Registry.update()`. Fix: point TMPDIR at a directory that exists; this
   module does that automatically when it can create it.

2. `could not load library "<cwd>/../lib/julia/sys.so"` -- Julia resolved its
   system image against the current directory instead of its own executable, so
   PythonCall could not precompile. Fix: import from a directory with no
   `../lib` to fall into; this module imports from the home directory.

If neither matches, run the import by itself in a job and read the first error,
not the last -- the `failed to precompile` lines are downstream of it:

    bsub -n 2 -gpu "num=1" -q gpu_a100 -W 1:00 -o probe.out \\
      "python -c 'import pysr; print(pysr.__version__)'"
"""
    with open(path, "w") as fh:
        fh.write(text)
    return path
