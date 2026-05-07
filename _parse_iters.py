import csv, statistics, os

ROOT = "/groups/saalfeld/home/allierc/GraphData/log/fly"
SLOT_PREFIX = "flyvis_noise_005_hidden_010_blank50_consensus_ngp_Claude"
NITER = 107000  # configured Niter for consensus_ngp_light

def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def window(rows, lo, hi, col):
    vals = []
    for r in rows:
        try:
            it = int(r['iteration'])
        except (KeyError, ValueError):
            continue
        if lo <= it < hi and r.get(col) not in (None, '', 'nan'):
            try:
                v = float(r[col])
                if v == v:
                    vals.append((it, v))
            except ValueError:
                pass
    return vals

def parse(slot):
    slot_dir = f"{ROOT}/{SLOT_PREFIX}_{slot:02d}/tmp_training"
    metrics = load(f"{slot_dir}/metrics.log")
    nnr = load(f"{slot_dir}/nnr_pearson.log")
    last_iter_m = max((int(r['iteration']) for r in metrics if r.get('iteration')), default=0)
    last_iter_n = max((int(r['iteration']) for r in nnr if r.get('iteration')), default=0)

    Niter = NITER
    p1 = window(metrics, 0, int(0.20*Niter), 'connectivity_r2')
    p1_end = window(metrics, int(0.18*Niter), int(0.20*Niter), 'connectivity_r2')
    p2 = window(metrics, int(0.20*Niter), int(0.50*Niter), 'connectivity_r2')
    p2_end = window(metrics, int(0.48*Niter), int(0.50*Niter), 'connectivity_r2')
    p3_dip = window(metrics, int(0.50*Niter), int(0.62*Niter), 'connectivity_r2')
    p3_final = window(metrics, int(0.95*Niter), Niter+1, 'connectivity_r2')

    # Fallback for p3_final when truncated: use the last 5% of *available* iters
    if not p3_final and last_iter_m > 0:
        p3_final = window(metrics, int(0.95*last_iter_m), last_iter_m+1, 'connectivity_r2')

    hid = window(nnr, int(0.90*Niter), Niter+1, 'hidden_pearson_mean')
    anc = window(nnr, int(0.90*Niter), Niter+1, 'anchor_pearson_mean')
    if not hid and last_iter_n > 0:
        hid = window(nnr, int(0.90*last_iter_n), last_iter_n+1, 'hidden_pearson_mean')
    if not anc and last_iter_n > 0:
        anc = window(nnr, int(0.90*last_iter_n), last_iter_n+1, 'anchor_pearson_mean')

    P1_peak = max(v for _, v in p1) if p1 else float('nan')
    P1_end = statistics.mean(v for _, v in p1_end) if p1_end else float('nan')
    P2_min = min(v for _, v in p2) if p2 else float('nan')
    P2_end = statistics.mean(v for _, v in p2_end) if p2_end else float('nan')
    P3_drop = (P2_end - min(v for _, v in p3_dip)) if (p3_dip and P2_end == P2_end) else float('nan')
    P3_final = statistics.mean(v for _, v in p3_final) if p3_final else float('nan')
    hid_final = statistics.mean(v for _, v in hid) if hid else float('nan')
    anc_final = statistics.mean(v for _, v in anc) if anc else float('nan')
    return dict(slot=slot, last_iter=last_iter_m, P1_peak=P1_peak, P1_end=P1_end,
                P2_min=P2_min, P2_end=P2_end, P3_drop=P3_drop, P3_final=P3_final,
                hid_final=hid_final, anc_final=anc_final,
                p3_final_window=(min(it for it,_ in p3_final), max(it for it,_ in p3_final)) if p3_final else None,
                hid_window=(min(it for it,_ in hid), max(it for it,_ in hid)) if hid else None)

def composite(r):
    P1_end = r['P1_end']; P2_end = r['P2_end']; P2_min = r['P2_min']
    P3_drop = r['P3_drop']; P3_final = r['P3_final']
    hid = r['hid_final']; anc = r['anc_final']
    s1 = P1_end if P1_end == P1_end else 0
    s2 = (P2_end - max(0, P2_end - P2_min)) if (P2_end == P2_end and P2_min == P2_min) else 0
    s3 = (P3_final - 0.5*max(0, P3_drop)) if (P3_final == P3_final and P3_drop == P3_drop) else 0
    valid_joint = [x for x in [P3_final, hid, anc] if x == x]
    sj = min(valid_joint) if valid_joint else 0
    return 0.30*s1 + 0.20*s2 + 0.30*s3 + 0.20*sj

for slot in [0,1,2,3]:
    r = parse(slot)
    c = composite(r)
    print(f"=== slot {slot:02d} === last_iter={r['last_iter']}")
    for k in ['P1_peak','P1_end','P2_min','P2_end','P3_drop','P3_final','hid_final','anc_final']:
        v = r[k]
        print(f"  {k}={v:.4f}" if v == v else f"  {k}=nan")
    print(f"  p3_final_window={r['p3_final_window']}  hid_window={r['hid_window']}")
    print(f"  composite={c:.4f}")
