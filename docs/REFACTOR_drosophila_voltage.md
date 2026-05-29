# REFACTOR_drosophila_voltage.md

**Headline: topology + task + voltage -> identifiable circuit.** Add a voltage-trace MSE loss inside the existing task training so the next paper reads as a direct continuation of `drosophila.tex`. **No new teacher/student class hierarchy** — the "teacher" is simply another already-trained `DrosophilaCxTaskRNN` / `DrosophilaCxTaskGNN` instance, and voltage targets are stored as `NeuronTimeSeries` (the existing on-disk format used by the data generators).

## A. Status quo

The drosophila CX pipeline is fully end-to-end direct supervision on heading-direction targets `(cos theta_hd, sin theta_hd)`. `DrosophilaCxTaskRNN` and `DrosophilaCxTaskGNN` (`src/connectome_gnn/models/drosophila_cx_task_rnn.py`, `drosophila_cx_task_gnn.py`) both consume the dict returned by `load_drosophila_cx_connectome(...)` (`connconstr_data.py:63-276`) and return `(y_hat, h_buf)` of shapes `(B,T,2)` and `(B,T,N)`. Loss assembly lives in `graph_trainer.py:2074-2100`: an `F.mse_loss(y_hat, y)` term plus regularisers (`coeff_cos_distance`, `coeff_norm_floor`, `coeff_tv_circular`, `coeff_W_L1`, `coeff_f_theta_diff`, `coeff_g_phi_diff`). The hidden state `h_buf` is never compared against anything — it only feeds priors. The 6-condition CV harness in `cv_runner.py:211-350` rotates seeds per fold (`seed`, `seed+1000`) across `cv00..cv09`. The repo already defines `NeuronState` and `NeuronTimeSeries` (`src/connectome_gnn/neuron_state.py:36-178, 182-310`) as the canonical containers for per-frame and full-rollout circuit observables (`voltage`, `stimulus`, `calcium`, `fluorescence`, ...); the flyvis / cx ODE generators already round-trip simulation traces through `NeuronTimeSeries`.

## B. Goal

`drosophila.tex` shows that **topology + task is insufficient** for full identifiability (the GNN converges in heading but f_theta / g_phi are not uniquely pinned). The next claim: **topology + task + a sparse set of voltage traces is sufficient.** Treat the "teacher" as another converged task model — concretely **Known-ODE +TV cv0** (`${GNN_OUTPUT_ROOT}/log/drosophila_cx/drosophila_cx_pi_epg_tv_cv0`), the bump-localised solution whose EPG-ring activity matches the canonical CX picture — whose `h_buf` rollouts are cached as `NeuronTimeSeries` on disk and supplied per training batch alongside the heading target. A single new loss term `L_voltage` is added to the existing task training. **No `TeacherNet`, no `VoltageTarget`, no separate distillation stage** — voltage supervision is just another regulariser, exactly like `coeff_tv_circular` is today.

## C. New entities

| Name | File | Public surface |
|---|---|---|
| `loss_voltage_mse` (helper, not a class) | `src/connectome_gnn/models/voltage_loss.py` (new, ~30 lines) | `def loss_voltage_mse(h_pred: Tensor, h_target: Tensor, neuron_ix: LongTensor, lam: float, use_sigmoid: bool=True) -> Tensor` — masked MSE in firing-rate space over the recorded subset |
| `voltage_metrics` (helpers, not a class) | `src/connectome_gnn/models/voltage_metrics.py` (new) | `def voltage_rmse(h_pred, h_gt, neuron_ix) -> float`; `def per_edge_recovery_r2(W_pred, W_gt, mask) -> float`; `def per_type_mi_alignment(h_pred, h_gt, neuron_types) -> dict[int, float]` |
| `teacher` (just a frozen TaskRNN, held in trainer state) | resolved by `models/graph_trainer.py` from `tc.teacher_config` | `teacher = create_model(...).requires_grad_(False).eval()` after `_load(tc.teacher_config)` returns the frozen Known-ODE +TV cv0 checkpoint; the trainer calls `teacher(trials.stimulus)` once per batch (no_grad) and writes its `h_buf` into `trials.voltage_target`. No new class, no pre-caching, no disk footprint. The teacher RNN is small (156 neurons, scalar $\tau$, sigmoid) so its forward pass is cheap next to the student's. |

**Reused entities (no changes needed unless noted):**
- `NeuronTimeSeries` (`neuron_state.py:182`) — already has `voltage: (T, N) float32`, exactly the right shape for cached teacher rollouts. Zarr round-trip is already there (`zarr_io.py`).
- `NeuronState` (`neuron_state.py:36`) — used inside the trainer for per-frame indexing into the cached `NeuronTimeSeries`.
- `DrosophilaCxTaskRNN` / `DrosophilaCxTaskGNN` — these *are* the teacher network. Loading a frozen, no-grad copy of an existing checkpoint and calling its `.forward(u)` produces `h_buf` of shape `(B, T, N)`. No `TeacherNet` wrapper needed.
- `TaskTrials` (`task_state.py`) — extended in (D) below to carry an optional pointer to the cached voltage; no new class.

## D. Modifications to existing code

- **`generators/task_state.py:33,35`.** Extend `TaskTrials` dataclass with `voltage_target: torch.Tensor | None = None  # (B,T,N) float32, sigmoid-space if so flagged`. Add `voltage_target` to `DYNAMIC_FIELDS` and `PI_FIELDS` so the existing `save` / `load` round-trip covers it.
- **`generators/utils.py:754-854` (`generate_path_integration_batch`).** No change. The batch generator stays oblivious to the teacher; the trainer is the one place that runs the teacher forward pass.
- **`generators/connconstr_data.py:63-276` (`load_drosophila_cx_connectome`).** No change.
- **`models/drosophila_cx_task_rnn.py` / `drosophila_cx_task_gnn.py`.** No `__init__` changes, no forward-pass changes. Add a thin `loss_voltage_mse(self, h_pred, h_target, neuron_ix, lam)` method mirroring the existing `loss_tv_circular` signature (TaskRNN line ~511; GNN line ~575). Both delegate to `voltage_loss.loss_voltage_mse`. This keeps the architectures structurally identical to today — the same class is the "teacher" when frozen and the "student" when trained.
- **`models/graph_trainer.py` (modify `_data_train_drosophila_cx_task` in place, do *not* fork a new function).** The 556-line function at line 1801 already supplies all the shared scaffolding — data load, three-group optimizer, scheduler, snapshot cadence, NaN guard, CV harness integration — and the zebrafish / cortex trainers at lines 2357 / 2390 call back into it. A sibling function would have to be kept bit-for-bit in sync. Two additions in-place. First, ahead of the training loop (where the student is built) resolve the teacher once:
  ```python
  teacher = None
  if float(getattr(tc, 'coeff_voltage_mse', 0.0)) > 0 and tc.teacher_config:
      teacher_cfg, _ = load_run_config(tc.teacher_config,
                                       explicit_output_root=False, task='train')
      teacher = _load_frozen(teacher_cfg, device)  # create_model + load_state_dict + eval()
      for p in teacher.parameters():
          p.requires_grad_(False)
      voltage_ix = _select_neurons(
          neuron_types=student.neuron_types,
          mode=tc.voltage_neuron_selection,
          fraction=tc.voltage_recorded_fraction,
          seed=tc.seed + 2000 + fold_index,
      ).to(device)
  ```
  Then inside the batch loop, right after `trials = generate_path_integration_batch(...)`:
  ```python
  if teacher is not None:
      with torch.no_grad():
          _, trials.voltage_target = teacher(trials.stimulus)
  ```
  And finally, in the loss assembly after `g_diff`:
  ```python
  cv = float(tc.coeff_voltage_mse)
  if cv > 0 and trials.voltage_target is not None:
      l_voltage = model.loss_voltage_mse(
          h_buf, trials.voltage_target, voltage_ix, cv,
      )
  else:
      l_voltage = u.new_zeros(())
  loss = mse + cosd + norm + tv + l1S + f_diff + g_diff + l_voltage
  ```
- **`config.py` (TrainingConfig).** Add: `coeff_voltage_mse: float = 0.0`; `teacher_config: str = ''` (the yaml name of the teacher); `voltage_neuron_selection: Literal['all','stratified_cell_type','random','per_glomerulus'] = 'stratified_cell_type'`; `voltage_recorded_fraction: float = 1.0`; `voltage_in_sigmoid_space: bool = True`. Defaults keep `coeff_voltage_mse = 0.0`; with this default, the three new blocks in `_data_train_drosophila_cx_task` are guarded out, the function is byte-equivalent to today's path, and the byte-equality contract from Section I is trivially satisfied.
- **`models/cv_runner.py:262-272`.** Forward the teacher fields through unchanged; per-fold randomisation of the recorded subset happens inside the trainer using `seed + 2000 + fold_index` (see snippet above) so the recorded-neuron noise is independent of `simulation.seed` and `training.seed`.
- **`models/drosophila_cx_eval.py:328-453` (`_save_training_snapshot`).** Add `voltage_rmse(h_buf, trials.voltage_target, voltage_ix)` to the metrics dict when `tc.coeff_voltage_mse > 0`.

## E. Training-pipeline changes

`L_voltage = lam_v * mean_{i in S} (sigmoid(h_pred_i) - sigmoid(h_target_i))^2` where `S` is the recorded subset. Apply on `sigmoid(h)` rather than raw h so the loss lives in firing-rate space (commensurable with TV and with the GNN's g_phi). Curriculum: weight tracks the existing `coeff_tail_loss` schedule, so voltage supervision turns on only after the heading curriculum has unrolled. Interaction with priors: `coeff_tv_circular` and `coeff_voltage_mse` are *not* mutually exclusive; a clean ablation needs both off, each alone, and both on. Monotonicity priors (`coeff_f_theta_diff`, `coeff_g_phi_diff`) stay independent of L_voltage. Grad-clip group stays the same; voltage gradients flow through `h_buf` into `W_rec` and into f_theta / g_phi for the GNN.

The teacher rollouts are computed **on the fly per batch**: the trainer holds a single frozen `DrosophilaCxTaskRNN` instance (Known-ODE $+$TV cv0, `${GNN_OUTPUT_ROOT}/log/drosophila_cx/drosophila_cx_pi_epg_tv_cv0/models/best_model_with_0_graphs_*.pt`) and calls `teacher(trials.stimulus)` under `no_grad()` immediately after each call to `generate_path_integration_batch(...)`. The result is stashed on `trials.voltage_target` and consumed by the loss. The teacher is small (156 neurons, scalar leak, sigmoid) so its forward pass adds milliseconds per step; pre-caching would cost $\sim$400 GB on disk for the same number of training batches and is not worth the complexity. The student gets fresh OU coverage every batch, matched 1-to-1 with the teacher's response to the same stimulus.

**Heading target stays the original ground truth.** The teacher does *not* supply `theta_hd` or any heading-loss term. Both teacher and student see the same `trials.stimulus`, and `trials.target = (cos\theta_{hd}, sin\theta_{hd})` continues to come from `generate_path_integration_batch`. The teacher's only contribution is `h_buf` -> `trials.voltage_target`. This way the paper's claim is sharply scoped: "voltage supervision recovers the teacher's *implementation* while keeping the I/O ground truth identical to the existing pipeline".

## F. Experimental matrix

Teacher: **Known-ODE $+$TV cv0** (`config/drosophila_cx/drosophila_cx_pi_epg_tv_cv0.yaml`, checkpoints at `${GNN_OUTPUT_ROOT}/log/drosophila_cx/drosophila_cx_pi_epg_tv_cv0/models/best_model_with_0_graphs_*.pt`, highest-epoch picked by `_load`).

Conditions to add (each x 10 CV folds):

1. **GNN no-TV, voltage 1%** — 2 neurons (1 EPG + 1 PEN), random per fold.
2. **GNN no-TV, voltage 10%** — ~15 neurons stratified across EPG/PEN/Delta7/PEG.
3. **GNN no-TV, voltage 100%** — full N=156 trace (upper bound).
4. **GNN $+$TV, voltage 10%** — TV + voltage composition test.
5. **Known-ODE +voltage 10%** — control: does the Known-ODE already saturate?
6. **GNN no-TV, voltage 10%, teacher = Known-ODE no-TV** — does teacher choice matter?

Negative controls (must appear in agentic-loop provenance per project rule): **GNN no-TV, voltage 10%, random teacher** (untrained ckpt) — expect L_voltage minimised, heading degraded.

## G. Identifiability metrics

Mirror `drosophila.tex` Section 4 (CV-mean tables) plus three new columns:

- **per-edge W_rec recovery R^2** between student and teacher `J_effective` after sign-locking (existing `nullspace.py:620-640` style).
- **per-cell-type MI alignment** between `sigmoid(h_pred)` and `sigmoid(h_target)` aggregated by `neuron_types` -> reuse `hd_mi_summary` axis labels for direct continuity.
- **embedding silhouette** on learned `a_i` (GNN only) clustered by ground-truth type.
- Existing metrics retained: `r_roll_1k`, `bump_FWHM`, `chi^2` on cos/sin, four-classes total (used in `cx_four_classes`).

Tables in the next paper should re-use the column layout of `fig_drosophila_cx_four_classes.py` so the reader sees voltage rows slotted beside the existing 6 conditions.

## H. First sanity-check experiment: parameter recovery from a Known-ODE teacher

Before any of the F-matrix conditions, the framework should be validated on a **fully known teacher**: train a `DrosophilaCxTaskGNN` student against the `DrosophilaCxTaskRNN` Known-ODE teacher at `${GNN_OUTPUT_ROOT}/log/drosophila_cx/drosophila_cx_pi_epg_tv_cv0` (`+TV` cv0, EPG-only readout, the bump-localised solution that motivates the paper's canonical alignment), with **100% voltage recording** so $L_{\mathrm{voltage}}$ is the only identifiability bottleneck. The teacher has three closed-form parameters that the GNN must recover:

1. **The scalar leak $\tau = 0.1$ s** — the teacher's drift is $-\hat h_i / \tau$ for every neuron. The GNN's drift is $f_\theta(\hat h_i, \mathbf{a}_i, m{=}0)$; if recovery works, $f_\theta$ should converge to a line of slope $-1/\tau \approx -10$ for every neuron over its operating range, irrespective of $\mathbf{a}_i$ (i.e. the per-neuron $\tau_i$ derived from the slope should collapse onto the scalar 0.1 s).
2. **The sigmoid firing-rate non-linearity $\sigma(h)$** — the teacher uses the standard sigmoid for the synaptic activation. The GNN's $g_\phi(\hat h_j, \mathbf{a}_j)$ should converge to that sigmoid, again irrespective of $\mathbf{a}_j$.
3. **The per-edge weights $\hat W_{ij}$** on the connectome support — the teacher has a single trained $\hat W^{\mathrm{rec}}$ matrix. The GNN learns $\hat W_{ij}$ at the same edge positions. Identifiability under voltage supervision means the GNN's $\hat W$ should match the teacher's on the connectome support.

These are the three quantities that the current paper explicitly says are *not* recovered under task-only supervision (Section "GNN with learned neuronal dynamics: the implementation is not uniquely identified by the task"). The sanity check measures whether voltage supervision closes that gap.

**In-training diagnostics (extend `fig_evolution_pi_gnn_*.png` rather than building from scratch):**

The existing `figures/drosophila_cx/fig_evolution.py` already plots the learned $f_\theta$ (panel k) and $g_\phi$ (panel l) every epoch. For the voltage-recovery runs, overlay the teacher's ground truth on those panels:

- **Panel k (drift / $\tau$ recovery)**: Per-neuron learned $f_\theta(\hat h_i, \mathbf{a}_i, m{=}0)$ in blue, the reference line $-\hat h / \tau$ for $\tau = 0.1$ s overlaid in red. Annotate the linear-fit slope of the learned $f_\theta$ in the panel title (`"slope = ${1/tau_eff:.2f} s^-1"`). Single-number summary: $\tau_{\mathrm{eff}} = -1 / \mathrm{mean}(\mathrm{slope}_i)$ vs. GT $\tau = 0.1$ s.
- **Panel l (signalling function / $\sigma$ recovery)**: Per-neuron learned $g_\phi(\hat h_j, \mathbf{a}_j)$ in blue, reference $\sigma(\hat h)$ overlaid in red. Annotate $\ell_2$ distance to sigmoid in the panel title.
- **New panel m (per-edge weight recovery)**: Scatter of teacher $\hat W^{\mathrm{rec}}_{ij}$ (GT, x-axis) vs. student $\hat W_{ij}$ (learned, y-axis) over the connectome support; identity line in red. Annotate Pearson $r$ and $R^2$ in the title. Mirror the per-cell-type colouring of the existing $\hat W^{\mathrm{rec}}$ visualisations.

Each of these three diagnostics is a one-shot scalar per epoch ($\tau_{\mathrm{eff}}$, sigmoid-$\ell_2$, $R^2_W$) that drops onto the same training-time `metrics.log` the trainer already writes, so the agentic loop can use them as recovery scores in addition to $r_{\mathrm{roll},1k}$.

**Pass/fail criterion for the sanity check (and headline result for the next paper if it passes):**

After the full curriculum, with $\sim$100% of neurons recorded, the GNN should converge to $\tau_{\mathrm{eff}} = 0.1 \pm 0.02$ s, sigmoid-$\ell_2 < 0.05$ over the visited operating range, and $R^2_W > 0.95$ on the connectome support — all across 10 CV folds. If any of the three fails, voltage supervision alone is *not* sufficient to identify the teacher's ODE, and the next paper has to localise which piece (e.g. $g_\phi$ requires monotonicity prior on top of voltage MSE) is missing. If all three pass, the F-matrix's "1% / 10% recording" rows become the quantitative recovery curve.

## I. Composition with zebrafish circuit registry

This refactor leans on the existing `NeuronState` / `NeuronTimeSeries` / `TaskTrials` triple that both biomodels already share, so no new abstractions are needed on the drosophila side to support voltage supervision. It does **not** introduce `circuits/` / `tasks/` / `io_mappings/` directories on the drosophila side yet — that is deferred until a second drosophila circuit (e.g. FB / NO) is needed. The voltage refactor sits orthogonally: the existing `DrosophilaCxTaskRNN` / `DrosophilaCxTaskGNN` keep their `_load_connectome` hook (lines 84-95 and 86-97), and a future registry move would simply wrap that hook. Byte-equality contract (golden gate B in the zebrafish doc) must hold for `coeff_voltage_mse=0`: a config with the new fields omitted produces identical training trajectories to the pre-refactor branch. Run gate D (omitted-vs-explicit-default) before merging.
