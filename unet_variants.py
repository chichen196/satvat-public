#!/usr/bin/env python3
"""Supplementary Tables S1 and S2: the three U-Net variants on the same folds, scored as in Table 3.

    python3 unet_variants.py --run 3class=runs/unet --run f2=runs/unet_f2 --run f2m=runs/unet_f2m \
                             --drop patient_179,patient_260 --out unet_variants.csv

Each --run is a run directory holding the sep_slices_unet.csv that aattct_unet.py writes, or that
CSV itself. --drop adds rows for each half without the subjects listed (Table S2). Measures, as in Table 3:
per-subject (summed slices) VAT and SAT %RMSE, Pearson r of VAT, median VAT error, subjects with
|VAT error| > 5%; and, pooled over all slices, F misassigned (SAT called VAT + VAT called SAT) and
F left unassigned, as % of F. Halves are odd- and even-numbered subjects, as in the paper.
"""
import argparse, csv, math, statistics as st
from collections import defaultdict

def load(path):
    rows = list(csv.DictReader(open(path, newline='')))
    S = defaultdict(lambda: defaultdict(float))
    for r in rows:
        s = S[r['subject']]
        for k in ('F_px', 'ref_sat', 'ref_vat', 'unet_sat', 'unet_vat', 'unet_none', 'unet_vat2sat', 'unet_sat2vat'):
            s[k] += float(r[k])
    return len(rows), S

def score(S, subjects):
    sub = [S[k] for k in subjects]
    ev = [100 * (s['unet_vat'] - s['ref_vat']) / s['ref_vat'] for s in sub]
    es = [100 * (s['unet_sat'] - s['ref_sat']) / s['ref_sat'] for s in sub]
    rm = lambda e: math.sqrt(sum(v * v for v in e) / len(e))
    F = sum(s['F_px'] for s in sub)
    return dict(n=len(sub), vat_rmse=rm(ev), sat_rmse=rm(es),
                r=st.correlation([s['ref_vat'] for s in sub], [s['unet_vat'] for s in sub]),
                med=st.median(ev), over5=sum(abs(v) > 5 for v in ev),
                mis=100 * sum(s['unet_sat2vat'] + s['unet_vat2sat'] for s in sub) / F,
                s2v=100 * sum(s['unet_sat2vat'] for s in sub) / F,
                v2s=100 * sum(s['unet_vat2sat'] for s in sub) / F,
                none=100 * sum(s['unet_none'] for s in sub) / F,
                clean=sum(s['unet_sat2vat'] + s['unet_vat2sat'] + s['unet_none'] == 0 for s in sub))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='name=run_dir')
    ap.add_argument('--drop', default='', help='comma-separated subjects for the sensitivity rows')
    ap.add_argument('--out', default='unet_variants.csv')
    a = ap.parse_args()
    runs = {}
    for spec in a.run:
        name, d = spec.split('=', 1)
        n, S = load(d if d.endswith('.csv') else f'{d}/sep_slices_unet.csv'); runs[name] = S
        print(f'{name}: {n} slices, {len(S)} subjects')
    subs = sorted(next(iter(runs.values())))
    assert all(sorted(S) == subs for S in runs.values()), 'runs cover different subjects'
    idn = lambda k: int(k.split('_')[-1])
    groups = {'odd': [k for k in subs if idn(k) % 2], 'even': [k for k in subs if not idn(k) % 2], 'all': subs}
    drop = [d for d in a.drop.split(',') if d]
    if drop:
        for g in ('odd', 'even', 'all'):
            groups[f'{g} without {"+".join(d.split("_")[-1] for d in drop)}'] = [k for k in groups[g] if k not in drop]
    out = []
    for g, ks in groups.items():
        for name, S in runs.items():
            m = score(S, ks); m.update(group=g, variant=name); out.append(m)
            print(f'{g:<22} {name:<7} n={m["n"]:<4} VAT%RMSE {m["vat_rmse"]:6.2f}  SAT%RMSE {m["sat_rmse"]:6.2f}  '
                  f'r {m["r"]:.4f}  med {m["med"]:+.2f}  >5% {m["over5"]:<3} mis {m["mis"]:.4f}  '
                  f'S2V {m["s2v"]:.4f}  V2S {m["v2s"]:.4f}  none {m["none"]:.4f}  clean {m["clean"]}')
    keys = ['group', 'variant', 'n', 'vat_rmse', 'sat_rmse', 'r', 'med', 'over5', 'mis', 's2v', 'v2s', 'none', 'clean']
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows({k: m[k] for k in keys} for m in out)

if __name__ == '__main__':
    main()
