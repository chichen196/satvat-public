#!/usr/bin/env python3
"""unet_overlay.py - overlays of the U-Net's SAT/VAT split for chosen or worst slices.

Reads the saved predictions of aattct_unet.py (pred/<subject>/<slice>.png, 0 bg / 1 SAT /
2 VAT) and draws them on the image with the same colours as aattct_sep.py: SAT agreed
orange, VAT agreed cyan, annotators' VAT called SAT red, annotators' SAT called VAT blue.
Only the fat mask F is coloured (F as in aattct_sep.py / aattct_unet.py), and a pixel of
F counts as VAT where the saved label map says 2.

    unet_overlay.py --root /path/to/AATTCT-IDS --run runs/unet --worst 12
    unet_overlay.py --root ... --run runs/unet --slices patient_260:FILE123,patient_196:FILE140

--worst N ranks slices by misassigned pixels from <run>/sep_slices_unet.csv.  The pred
map is the argmax over three classes, so it can differ from the scored split (VAT score
vs SAT score only) on fat pixels the network called background; those are drawn magenta.
"""
import argparse, csv, re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--run', default='runs/unet')
ap.add_argument('--worst', type=int, default=0)
ap.add_argument('--slices', default='', help='comma list of patient_NNN:FILEnnn')
ap.add_argument('--out', default='')
ap.add_argument('--fat-hi', type=float, default=79)
ap.add_argument('--body', type=float, default=40)
a = ap.parse_args()
root, run = Path(a.root).expanduser(), Path(a.run)
out = Path(a.out) if a.out else run / 'overlay'
out.mkdir(parents=True, exist_ok=True)

want = [tuple(x.split(':', 1)) for x in a.slices.split(',') if ':' in x]
if a.worst:
    with open(run / 'sep_slices_unet.csv') as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: -(int(r['unet_vat2sat']) + int(r['unet_sat2vat'])))
    want += [(r['subject'], r['slice']) for r in rows[:a.worst]]
if not want:
    raise SystemExit('nothing to draw: give --worst N or --slices')


def one(d, stem):
    return next(iter(sorted(d.glob(stem + '.*'))), None)


def readlab(p):
    m = np.array(Image.open(p))
    return m.any(-1) if m.ndim == 3 else m > 0


import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
for subj, sid in want:
    ip = one(root / 'Image' / 'Extracted' / subj, sid)
    sp = one(root / 'Label' / 'Subcutaneous' / subj, sid)
    vp = one(root / 'Label' / 'Visceral' / subj, sid)
    pp = run / 'pred' / subj / f'{sid}.png'
    if None in (ip, sp, vp) or not pp.exists():
        print(f'{subj}:{sid}: missing image, label or prediction'); continue
    g = np.array(Image.open(ip).convert('L')); s = readlab(sp); v = readlab(vp)
    b = ndi.binary_fill_holes(ndi.binary_closing((g > a.body) | s | v, iterations=3))
    cc, n = ndi.label(b)
    if n:
        b = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
    F = b & (g <= a.fat_hi) & (s | v)
    lab = np.array(Image.open(pp))
    RS, RV = F & s, F & v
    PV, PS, P0 = F & (lab == 2), F & (lab == 1), F & (lab == 0)
    gray = np.dstack([g / 255.0] * 3); ov = gray.copy()
    for m, c in ((RS & PS, (1, .55, 0)), (RV & PV, (0, .8, 1)), (RV & PS, (1, 0, 0)), (RS & PV, (0, 0, 1)), (P0, (1, 0, 1))):
        ov[m] = .35 * gray[m] + .65 * np.array(c)
    rv = int(RV.sum()) or 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    ax[0].imshow(gray); ax[1].imshow(ov)
    for x in ax: x.axis('off')
    ax[0].set_title(f'{subj} {sid}', fontsize=9)
    ax[1].set_title(f'VAT ref {rv} unet {int(PV.sum())}  red: ref VAT->SAT {int((RV & PS).sum())}  '
                    f'blue: ref SAT->VAT {int((RS & PV).sum())}  magenta: bg {int(P0.sum())}', fontsize=8)
    fig.tight_layout(); fig.savefig(out / f'{subj}_{sid}.png', dpi=90); plt.close(fig)
    print(f'-> {out}/{subj}_{sid}.png')
