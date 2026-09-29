"""The hidden-trace warm-up (zeros first, learned traces after) happens once per run.

The trainer rebuilds the schedule at every epoch with a per-epoch iteration
count, so a run of 20 epochs (recurrent training, one per horizon) used to
silence the learned traces again at the start of each of them.
"""
from types import SimpleNamespace

from connectome_gnn.models.training_utils import init_hidden_injection_schedule


def _training(**kw):
    return SimpleNamespace(warmup_inject_nnr_iter_frac=0.2,
                           warmup_inject_nnr_ramp_iter_frac=0.05,
                           lr_damping_factor=100.0, **kw)


def test_the_first_epoch_warms_up():
    hs = init_hidden_injection_schedule(_training(), 1000, first_epoch=True)
    assert hs.warmup_iter == 200 and hs.damping_active


def test_later_epochs_inject_from_the_first_iteration():
    hs = init_hidden_injection_schedule(_training(), 1000, first_epoch=False)
    assert hs.warmup_iter == 0 and not hs.damping_active


def test_the_default_keeps_the_old_single_epoch_behaviour():
    assert init_hidden_injection_schedule(_training(), 1000).warmup_iter == 200
