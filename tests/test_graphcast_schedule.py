"""GraphCast-style LR schedule, global clipping and short rollout tail.

Three knobs added after reading GraphCast's curriculum (supplement sec 4.3-4.5):
its 311,000 updates are 96.5% one-step under a monotone decay, with rollout only
as a 3.5% tail fine-tune at an LR ~3300x below peak. Our Task 1 rollout arms
instead ramped K across the whole run at constant LR, which is a precise
explanation for "reached 0.983 fast, then declined to 0.961".

  training.lr_scheduler = 'graphcast'      warmup -> ONE half-cosine -> flat tail
  training.grad_clip_norm                  global clip over ALL parameters
  training.rollout_tail_iters_per_epoch    cap on K>1 epochs, so the tail is short
"""
import math

import pytest
import torch

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.training_utils import build_lr_scheduler, planned_total_updates
from connectome_gnn.utils import config_path

pytestmark = pytest.mark.tier2

CFG = "fly/flyvis_noise_005_gc_graphcast_cv00.yaml"
CFG_CTL = "fly/flyvis_noise_005_gc_control_cv00.yaml"


def _cfg(name):
    return NeuralGraphConfig.from_yaml(config_path(name))


def _lrs(cfg, n):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=1.0)
    sched = build_lr_scheduler(opt, cfg)
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        sched.step()
    return out


class TestGraphcastSchedule:
    def test_warmup_then_monotone_decay_then_flat_tail(self):
        cfg = _cfg(CFG)
        t = cfg.training
        total = planned_total_updates(cfg)
        lrs = _lrs(cfg, total)
        w = t.lr_scheduler_warmup_iters
        decay_end = int(round(total * t.lr_scheduler_decay_frac))

        assert lrs[0] < 1e-6, "must start at ~0"
        assert lrs[w] == pytest.approx(1.0, rel=1e-6), "must reach peak at end of warmup"
        # strictly non-increasing after warmup -- the whole point vs
        # linear_warmup_cosine, which uses warm RESTARTS and sawtooths back up
        after = lrs[w:]
        assert all(b <= a + 1e-12 for a, b in zip(after, after[1:]))
        assert lrs[-1] == pytest.approx(t.lr_scheduler_tail_ratio, rel=1e-9)
        assert lrs[decay_end] == pytest.approx(t.lr_scheduler_tail_ratio, rel=1e-6)

    def test_decay_is_a_half_cosine(self):
        cfg = _cfg(CFG)
        t = cfg.training
        total = planned_total_updates(cfg)
        lrs = _lrs(cfg, total)
        w = t.lr_scheduler_warmup_iters
        decay_end = int(round(total * t.lr_scheduler_decay_frac))
        tail = t.lr_scheduler_tail_ratio
        for frac in (0.25, 0.5, 0.75):
            step = w + int((decay_end - w) * frac)
            # expectation from the ACTUAL integer step, not the nominal fraction:
            # int() truncation shifts p by up to one step, which at this scale is a
            # ~1e-7 relative change in lr and would otherwise look like a mismatch
            p = (step - w) / (decay_end - w)
            want = tail + (1 - tail) * 0.5 * (1 + math.cos(math.pi * p))
            assert lrs[step] == pytest.approx(want, rel=1e-9)

    def test_control_arm_is_constant(self):
        cfg = _cfg(CFG_CTL)
        assert cfg.training.lr_scheduler == "none"
        lrs = _lrs(cfg, 500)
        assert all(x == pytest.approx(1.0) for x in lrs)


class TestRolloutTail:
    def test_tail_is_a_small_fraction_of_updates(self):
        """The rollout phase must be a fine-tune, not the bulk of training."""
        cfg = _cfg(CFG)
        t = cfg.training
        total = planned_total_updates(cfg)
        sim = cfg.simulation
        niter = int(sim.n_frames * t.data_augmentation_loop // t.batch_size * 0.2)
        n_one_step = sum(1 for k in t.rollout_horizon_schedule if k == 1) * niter
        tail = total - n_one_step
        assert 0 < tail / total < 0.05, f"tail is {tail/total:.1%} of updates, want <5%"
        # GraphCast's own split is 3.5%
        assert tail / total == pytest.approx(0.035, abs=0.005)

    def test_lr_decay_ends_where_the_tail_begins(self):
        """The cosine must reach its floor exactly when K starts ramping.

        Otherwise the rollout phase runs at a still-high LR, which is the Task 1
        failure mode this arm exists to avoid.
        """
        cfg = _cfg(CFG)
        t = cfg.training
        total = planned_total_updates(cfg)
        sim = cfg.simulation
        niter = int(sim.n_frames * t.data_augmentation_loop // t.batch_size * 0.2)
        n_one_step = sum(1 for k in t.rollout_horizon_schedule if k == 1) * niter
        decay_end = int(round(total * t.lr_scheduler_decay_frac))
        assert abs(decay_end - n_one_step) / total < 0.01

    def test_cap_only_applies_to_K_gt_1(self):
        cfg = _cfg(CFG)
        t = cfg.training
        sim = cfg.simulation
        niter = int(sim.n_frames * t.data_augmentation_loop // t.batch_size * 0.2)
        cap = t.rollout_tail_iters_per_epoch
        assert cap > 0
        # K=1 epochs keep the full Niter; K>1 epochs are capped
        assert planned_total_updates(cfg) == (
            sum(niter if k == 1 else min(max(1, niter // k), cap)
                for k in t.rollout_horizon_schedule))

    def test_no_cap_reproduces_old_behaviour(self):
        cfg = _cfg(CFG)
        cfg.training.rollout_tail_iters_per_epoch = 0
        t, sim = cfg.training, cfg.simulation
        niter = int(sim.n_frames * t.data_augmentation_loop // t.batch_size * 0.2)
        assert planned_total_updates(cfg) == sum(
            max(1, niter // k) for k in t.rollout_horizon_schedule)


class TestGlobalClip:
    def test_clips_every_parameter_not_just_W(self):
        """grad_clip_W touches only model.W; grad_clip_norm must bound the whole step."""
        torch.set_default_device("cpu")
        W = torch.nn.Parameter(torch.zeros(4))
        other = torch.nn.Parameter(torch.zeros(4))
        W.grad = torch.full((4,), 100.0)
        other.grad = torch.full((4,), 100.0)
        torch.nn.utils.clip_grad_norm_([W, other], max_norm=32.0)
        total = torch.cat([W.grad, other.grad]).norm().item()
        assert total == pytest.approx(32.0, rel=1e-5)
        assert other.grad.norm().item() < 100.0, "non-W params must be clipped too"

    def test_arms_configure_it_as_intended(self):
        assert _cfg(CFG_CTL).training.grad_clip_norm == 0.0
        assert _cfg("fly/flyvis_noise_005_gc_anneal_cv00.yaml").training.grad_clip_norm == 0.0
        assert _cfg("fly/flyvis_noise_005_gc_annealclip_cv00.yaml").training.grad_clip_norm == 32.0
        assert _cfg(CFG).training.grad_clip_norm == 32.0
