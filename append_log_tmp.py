"""Iter 161-180: append Block 4.2 log entry to cortex_unique_matrix analysis log."""

LOG = '/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_cortex_unique_matrix/cortex_unique_matrix_Claude_analysis.md'

log_entry = '''

---

## Iter 161-180 (block 4, batch 4.2): EXPLORATION — 3-epoch plateau at 2e-3 then geometric x0.5/epoch
Mutation (uniform on all 20 slots): lr_W_rec_schedule = [2e-3, 2e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.13e-5, 1.56e-5]; ED schedule unchanged (decoupled_low [5e-4, 2e-4, 1e-4 x 8]); rate_L2=1e-2, W_L2=0, noise=0, grad_clip=2.0, batch_size=128, sigma=true.
Hypothesis (H11): "3-epoch plateau at 2e-3 then geometric x0.5/epoch gives 1024 more high-LR exploration time and raises ep-3 ceiling above 2.3's 0.872."

### n_units = 256 (slots 0-6) - all completed 10 epochs (~28 min)
| Slot | Seed sim/train | r2_best (traj peak) | r2_final | dir_acc_end |
|---:|:---:|:---:|:---:|:---:|
| 0 | 161000/161500 | 0.839 | 0.839 | 0.609 |
| 1 | 162001/162501 | 0.871 | 0.799 | 0.547 |
| 2 | 163002/163502 | 0.853 | 0.708 | 0.531 |
| 3 | 164003/164503 | 0.899 | 0.807 | 0.641 |
| 4 | 165004/165504 | 0.856 | 0.752 | 0.625 |
| 5 | 166005/166505 | **0.939** | 0.848 | 0.625 |
| 6 | 167006/167506 | 0.901 | 0.856 | 0.656 |

**Aggregate trajectory-peak: mean = 0.880, std = 0.034, ceiling = 0.939** (n=7) - NEW 256 records on both mean AND ceiling.
Final-r2: mean = 0.801, std = 0.054, ceiling = 0.856 - final-r2 mean trails 4.1's 0.829 because plateau-end LR drop spikes noise (slot 5 ep10 r2=0.848 vs r2_best=0.939 -> 0.091 final-vs-peak gap; same pattern in slots 1, 2, 4).

### n_units = 512 (slots 7-13) - ALL FAILED, no metrics written
All 7 slots' `*_analysis.log` are empty (1 line), no `r2_trajectory/iter_16{8,9}.log` or iter_17{0..3}.log produced. Same divergence-style failure as Block 1's 1024 collapse - extended 3-epoch plateau at 2e-3 was unstable at 512.

### n_units = 1024 (slots 14-19) - ALL FAILED, no metrics written
Same as 512: empty analysis logs, no trajectory files. The 3-epoch plateau at 2e-3 is fatal at 1024 (replays the Block 1 collapse mode).

Per-size summary (mean +/- std across seeds at each n_units):
  n_units=256:  mean=0.880  std=0.034  ceiling=0.939  (NEW RECORDS)
  n_units=512:  N/A - diverged
  n_units=1024: N/A - diverged

Late-stage check (256 only): slot 5 hits 0.939 at ep10 with motor_max 0.843 (healthy gain). Slot 6 final r2=0.856 (best final this batch). Slot 3 ceiling 0.899.
Saturation diagnosis (256 only this batch): cannot read off-curve; 256 alone improved.

Verdict: **H11 PARTIAL - strongly supported at 256 (mean +0.051 over best prior, ceiling +0.013), strongly falsified at 512/1024 (catastrophic divergence).** The extended plateau gives 256 more time to refine W with high LR but pushes 512/1024 past their stability margin. Same pattern as Block 1 collapse: long high-LR window destabilises large recurrent matrices.

Next mutation (Block 5.1, iter 181-200): **CV final on Block 4.1 cosine winner** - most robust config across all three sizes (256 mean 0.829 / 512 mean 0.873 record / 1024 ep-3 ceiling 0.833, no divergence). All 20 slots = identical 4.1 cosine config; pipeline-forced seeds give n=7 x 256 + 7 x 512 + 6 x 1024 = 20 fresh seeds for the variance estimate.
'''

with open(LOG, 'a') as f:
    f.write(log_entry)
print(f'appended {len(log_entry)} chars to {LOG}')
