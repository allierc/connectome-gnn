"""Iter 181-200 Block 5.1 CV: append Block 4.2 log entry and rewrite 20 YAML configs
to Block 4.1 cosine winner. Slot n_units stay fixed (256 for 0-6, 512 for 7-13,
1024 for 14-19). All other knobs match 4.1 (decoupled_low ED, rate_L2=1e-2,
W_L2=0, noise=0, grad_clip=2.0, batch_size=128, sigma=true)."""

import os
import re

LOG = '/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_cortex_unique_matrix/cortex_unique_matrix_Claude_analysis.md'

log_entry = '''

---

## Iter 161-180 (block 4, batch 4.2): EXPLORATION - 3-epoch plateau at 2e-3 then geometric x0.5/epoch
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
All 7 slots' `*_analysis.log` are empty (1 line), no r2_trajectory iter files produced for 168-174. Same divergence-style failure as Block 1's 1024 collapse - extended 3-epoch plateau at 2e-3 was unstable at 512.

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

if os.path.exists(LOG):
    with open(LOG, 'a') as f:
        f.write(log_entry)
    print(f'appended {len(log_entry)} chars to log')
else:
    print(f'LOG file missing: {LOG}')


# ----- patch 20 configs to Block 4.1 cosine winner -----
CFG_DIR = '/groups/saalfeld/home/allierc/GraphData/config/cortex'

# Block 4.1 cosine W_rec schedule (verified from iter_141_slot_00.yaml).
W_REC_COSINE = [0.002, 0.00194, 0.00176, 0.00149, 0.00117, 0.000822, 0.000497, 0.000233, 6.0e-05, 1.0e-05]
W_ED_DEC_LOW = [0.0005, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

# Scalar carry-forwards from 2.3 base (idempotent).
TARGETS = {
    'coeff_rate_L2': '0.01',
    'grad_clip_W': '2.0',
    'coeff_W_L2': '0.0',
    'noise_recurrent_level': '0.0',
    'lr': '0.001',
    'n_epochs': '10',
    'batch_size': '128',
    'data_augmentation_loop': '80',
    'recurrent_activation': 'tanh',
    'readout_uses_sigma': 'true',
    'w_init_mode': 'randn_scaled',
    'w_init_scale': '0.5',
    'input_proj': 'matrix',
    'output_proj': 'matrix',
}


def fmt(v):
    """Format scalar so YAML round-trips exactly like the original."""
    s = f'{v:g}'
    # ruamel-style: scientific must have e-NN with leading zero for negatives <0.001
    if 'e' in s and v == 6e-05:
        s = '6.0e-05'
    if 'e' in s and v == 1e-05:
        s = '1.0e-05'
    return s


def patch_yaml(path):
    with open(path, 'r') as f:
        content = f.read()
    orig_len = len(content)
    changes = []

    for key, val in TARGETS.items():
        if key == 'lr':
            pat = re.compile(r'^(  )lr:\s*[^\n#]*$', re.MULTILINE)
        else:
            pat = re.compile(rf'^(\s+){re.escape(key)}:\s*[^\n#]*$', re.MULTILINE)
        new_content, n = pat.subn(lambda m, k=key, v=val: f'{m.group(1)}{k}: {v}', content)
        if n > 0 and new_content != content:
            content = new_content
            changes.append(f'{key}({n})')

    pat_rec = re.compile(
        r'^(  )lr_W_rec_schedule:\s*\n(?:  - [\d.eE+-]+\s*\n){1,30}',
        re.MULTILINE,
    )
    rec_block = '  lr_W_rec_schedule:\n' + ''.join(f'  - {fmt(v)}\n' for v in W_REC_COSINE)
    new_content, n = pat_rec.subn(rec_block, content)
    if n > 0:
        content = new_content
        changes.append(f'lr_W_rec_schedule({n})')

    pat_ed = re.compile(
        r'^(  )lr_W_ED_schedule:\s*\n(?:  - [\d.eE+-]+\s*\n){1,30}',
        re.MULTILINE,
    )
    ed_block = '  lr_W_ED_schedule:\n' + ''.join(f'  - {fmt(v)}\n' for v in W_ED_DEC_LOW)
    new_content, n = pat_ed.subn(ed_block, content)
    if n > 0:
        content = new_content
        changes.append(f'lr_W_ED_schedule({n})')

    with open(path, 'w') as f:
        f.write(content)
    return changes, orig_len, len(content)


for i in range(20):
    path = f'{CFG_DIR}/cortex_unique_matrix_Claude_{i:02d}.yaml'
    if not os.path.exists(path):
        print(f'  slot {i:02d} MISSING')
        continue
    changes, before, after = patch_yaml(path)
    print(f'  slot {i:02d}: {len(changes)} keys patched ({before}->{after} bytes) {",".join(changes)}')

print('DONE patching all 20 configs.')
