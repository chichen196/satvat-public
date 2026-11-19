#!/usr/bin/env python3
"""unet_rescore.py - score saved U-Net predictions on a narrower fat mask, without rerunning.

aattct_unet.py (--mode f2/f2m) writes pred/<subject>/<slice>.png: 0 background, 1 SAT,
2 VAT on every pixel of its fat mask F (grey <= 79).  A stricter fat definition
(e.g. grey <= 40) gives a subset of F, so the same predictions can be scored on it:

    F' = body & grey <= --fat-hi & (SAT label | VAT label)      (body as aattct_unet.py)

The network is unchanged; only the evaluation set shrinks.  Output: a summary in the
layout of totalseg_aattct.py's ts_summary.txt (per-subject %RMSE from summed slices,
odd/even halves by subject number), plus unet_rescore_subjects.csv.

    python3 unet_rescore.py --root /data/LCClinic/fat/AATTCT-IDS --run runs/unet_f2m --fat-hi 40
"""
import argparse, csv, re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--run', required=True, help='aattct_unet.py --out folder (holds pred/)')
ap.add_argument('--fat-hi', type=float, default=40)
ap.add_argument('--body', type=float, default=40)
ap.add_argument('--out', default='', help='summary file (default <run>/unet_rescore_fh<fat-hi>.txt)')
a = ap.parse_args()
root, run = Path(a.root).expanduser(), Path(a.run)
pred_dir = run / 'pred'
if not pred_dir.is_dir():
    raise SystemExit(f'{pred_dir} not found (was the run made with --no-save-pred?)')


def readlab(p):
    m = np.array(Image.open(p))
    return m.any(-1) if m.ndim == 3 else m > 0


def find(d, stem):
    for p in d.glob(stem + '.*'):
        return p
    return None


def body_mask(g, s, v):
    b = ndi.binary_fill_holes(ndi.binary_closing((g > a.body) | s | v, iterations=3))
    cc, n = ndi.label(b)
    if n:
        b = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
    return b


def snum(s):
    d = re.findall(r'\d+', s)
    return int(d[-1]) if d else 0


subj = []
for sd in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
    tot = dict(S=0, V=0, PS=0, PV=0, F=0, s2v=0, v2s=0, nei=0, outside=0)
    for pp in sorted(sd.glob('*.png')):
        ip = find(root / 'Image' / 'Extracted' / sd.name, pp.stem)
        sp = find(root / 'Label' / 'Subcutaneous' / sd.name, pp.stem)
        vp = find(root / 'Label' / 'Visceral' / sd.name, pp.stem)
        if ip is None or sp is None or vp is None:
            continue
        g = np.array(Image.open(ip).convert('L'))
        s, v = readlab(sp), readlab(vp)
        lab = np.array(Image.open(pp))
        F = body_mask(g, s, v) & (g <= a.fat_hi) & (s | v)
        S, V = F & s, F & v
        PS, PV = F & (lab == 1), F & (lab == 2)
        tot['S'] += int(S.sum()); tot['V'] += int(V.sum())
        tot['PS'] += int(PS.sum()); tot['PV'] += int(PV.sum()); tot['F'] += int(F.sum())
        tot['s2v'] += int((S & PV).sum()); tot['v2s'] += int((V & PS).sum())
        tot['nei'] += int((F & (lab == 0)).sum())          # should be 0: F' lies inside the run's F
        tot['outside'] += int((F & (lab == 0)).sum())
    if tot['F']:
        subj.append(dict(subject=sd.name, half='odd' if snum(sd.name) % 2 else 'even', **tot,
                         vat_err_pct=100 * (tot['PV'] - tot['V']) / tot['V'] if tot['V'] else np.nan,
                         sat_err_pct=100 * (tot['PS'] - tot['S']) / tot['S'] if tot['S'] else np.nan))
if not subj:
    raise SystemExit('nothing scored')

with open(run / f'unet_rescore_fh{a.fat_hi:g}_subjects.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(subj[0])); w.writeheader(); w.writerows(subj)
lines = [f'U-Net ({run}) rescored on fat = grey <= {a.fat_hi:g}; predictions unchanged',
         'half    n   VAT%RMSE  SAT%RMSE  F_mis%  SATcalledVAT%F  VATcalledSAT%F  neither%F  VATerr>5%  zero-mis']
for half in ('odd', 'even', 'all'):
    ss = [x for x in subj if half == 'all' or x['half'] == half]
    if not ss:
        continue
    ve = np.array([x['vat_err_pct'] for x in ss], float); se = np.array([x['sat_err_pct'] for x in ss], float)
    F = sum(x['F'] for x in ss)
    s2v, v2s, nei = (sum(x[k] for x in ss) for k in ('s2v', 'v2s', 'nei'))
    lines.append(f'{half:5s} {len(ss):4d} {np.sqrt(np.nanmean(ve ** 2)):9.2f} {np.sqrt(np.nanmean(se ** 2)):9.2f} '
                 f'{100 * (s2v + v2s + nei) / F:7.3f} {100 * s2v / F:15.3f} {100 * v2s / F:15.3f} '
                 f'{100 * nei / F:10.3f} {int(np.sum(np.abs(ve) > 5)):10d} '
                 f'{sum(1 for x in ss if x["s2v"] + x["v2s"] + x["nei"] == 0):9d}')
outside = sum(x['outside'] for x in subj)
if outside:
    lines.append(f'warning: {outside} pixels of the new fat mask had no prediction (not inside the run\'s F?)')
txt = '\n'.join(lines) + '\n'
(Path(a.out) if a.out else run / f'unet_rescore_fh{a.fat_hi:g}.txt').write_text(txt)
print(txt, end='')
