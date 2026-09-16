"""A progress bar for `MultiTaskSolver.train` that does not reimplement its loop.

WHY A WRAPPER AND NOT A COPY OF THE LOOP. The obvious way to get a bar is to
subclass the solver and paste its `train` body with a tqdm around the dataloader.
That would duplicate about eighty lines of the exact training regime
REPRODUCING.md is asking us to match, and any later divergence between our copy
and flyvis's original would be invisible -- a reproduction that quietly stopped
reproducing.

THE SEAM USED INSTEAD is `Task.loss`, which the training loop calls once per task
per batch (solver.py:336). Replacing the bound method on the task INSTANCE gives
a per-iteration callback carrying the loss value itself, which is the one number
worth watching, and leaves flyvis's loop untouched. `solver.iteration` and the
optimizer's learning rate are read off the solver for the rest of the postfix.
"""

from __future__ import annotations

import sys
from collections import deque

from tqdm import tqdm

__all__ = ["TrainingProgress"]


class TrainingProgress:
    """Context manager showing iteration, loss, learning rate and iterations/second.

    Args:
        solver: an initialised `flyvis.solver.MultiTaskSolver`.
        ncols: bar width in characters.
        window: how many recent iterations the displayed loss averages over. A
            single batch's loss on this task is far too noisy to read, so the bar
            shows a running mean; `smooth` in the postfix names the window so the
            number is never mistaken for an instantaneous value.
    """

    def __init__(self, solver, ncols: int = 150, window: int = 50) -> None:
        self.solver = solver
        self.ncols = ncols
        self.window = window
        self._recent: deque[float] = deque(maxlen=window)
        self._bar: tqdm | None = None
        self._original_loss = None
        # Only the FIRST task ticks the bar. With one task ('flow') that is every
        # call, but a two-task config would otherwise advance the bar twice per
        # iteration and report a rate that is double the real one.
        self._tick_task = next(iter(solver.task.dataset.tasks))

    def __enter__(self) -> "TrainingProgress":
        total = int(self.solver.task.n_iters)
        # ON A CLUSTER THERE IS NO TTY, and tqdm then emits every refresh as its
        # own line: a 250,000-iteration run would write hundreds of thousands of
        # lines into the bsub output file. Throttling to one line a minute keeps
        # the log readable and still shows the run progressing, which is the only
        # thing it is for once nobody is watching it live.
        interactive = sys.stderr.isatty()
        self._bar = tqdm(
            total=total,
            initial=int(self.solver.iteration),
            ncols=self.ncols,
            desc=f"train {self.solver.dir.path.name}",
            unit="it",
            mininterval=0.1 if interactive else 60.0,
        )
        self._original_loss = self.solver.task.loss
        self.solver.task.loss = self._loss  # shadows the bound method on the instance
        return self

    def __exit__(self, *exc) -> None:
        if self._original_loss is not None:
            self.solver.task.loss = self._original_loss
        if self._bar is not None:
            self._bar.close()
        return None

    def _loss(self, input, target, task: str, **kwargs):
        out = self._original_loss(input, target, task, **kwargs)
        if task == self._tick_task and self._bar is not None:
            # `.item()` synchronises, but the training loop already moves this
            # same scalar to the cpu every iteration, so the bar adds no stall
            # the loop was not paying anyway.
            self._recent.append(float(out.detach()))
            mean = sum(self._recent) / len(self._recent)
            # The KEY SAYS IT IS AN AVERAGE, and over how many iterations. Named
            # `smooth` it read like a training hyperparameter beside `lr`, which
            # is flyvis's real one; this is only how the bar displays the loss.
            self._bar.set_postfix(
                **{f"loss_avg{len(self._recent)}": f"{mean:.4f}"},
                lr=f"{self.solver.optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )
            self._bar.update(1)
        return out
