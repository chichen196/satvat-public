#!/usr/bin/env python3
"""chaos_mri_screen.py - screen MRI volumes for the MRI arm (CHAOS T1-DUAL in-phase; Oct 2026).

Derived from amos_mri_screen.py.  The 2006 method needs axial slices on which fat is bright
(not suppressed), the whole trunk cross-section inside the field of view, and abdominal
slices below the lungs.  This script measures what it can from the images (and organ labels
where they exist) and draws pictures for a visual call.  Nothing is changed on disk.

    chaos_mri_screen.py --root /path/to/CHAOS --out runs/chaos_screen
    chaos_mri_screen.py --root ... --cases 22,30,33 --out runs/chaos_try

--root is searched recursively for <prefix>_NNNN[_suffix].nii[.gz] (default prefix chaos;
dcm2niix's _e2 is fine); a file under a folder whose name starts with "image" is an image,
under "label" or "mask" a label map with AMOS codes (1 spleen, 2/3 kidneys, 4 gallbladder,
6 liver, 8 aorta).  --prefix amos --min-id 500 screens AMOS as before.

Per volume, after reorientation to RAS (axial slice = third axis):
  geometry  canonical shape and spacing; acquisition plane = the stored axis with the
            coarsest spacing (axial / coronal / sagittal; '3D' when near isotropic)
  body      per axial slice, median-filtered (--median 5, against speckle and ghosting):
            intensity > --body-frac x the volume's 99.5th percentile, closed, holes filled; the trunk is separated from the arms by mri_trunk.trunk_mask
            (notches in front of and behind an arm pressed against the flank, cut tip to tip;
            an opening of --arm-open mm as fallback, kept only for pieces with muscle in them).
            arm_slices = level slices with any arm (free or cut off); arm_cut_slices = those
            where the arm touched the trunk and a notch cut was made; arms = median arm
            area / trunk area over the level slices (so arms on a few slices leave it 0).  'cut' = the TRUNK runs along
            an image edge for more than --edge-frac of that edge (arms cut by the field of
            view do not count); 'arms' = arm area over 5% of the trunk
  lungs     a slice shows lung when the trunk holds dark regions (below the body
            threshold or --lung-frac x the trunk median, whichever is higher; each
            >= --lung-min mm^2) on BOTH sides of the midline, in its
            posterior half or (higher up, where the lungs fill the chest) of 10 x that size;
            the run starts at the highest such slice in the upper 40% of the stack (the top
            slices may show none) and is followed down, one-slice gaps allowed the lung run is followed down from the top of the stack (one-slice gaps
            allowed), so gas in the colon lower down does not count
  level     slices holding a kidney label (2, 3) where a label map exists, else every slice
            below the lungs (the abdomen the stack covers: abd_mm = those slices x spacing)
  contrast  medians inside the liver (6), spleen (1), gallbladder (4), aorta (8) and
            kidney (2, 3) labels; fat = 90th percentile of the body's outer rim (2-12 mm inside the
            skin) on the level slices; each as a ratio to the liver, or without a label map
            to 'lean' = the 30th percentile of the trunk above the body threshold (lean
            tissue even in a subject whose trunk is half fat; the body median is not)
  verdict   reject: fat not bright (fat/liver < --min-fat 1.2; fat/lean < --min-fat-lean 1.5 is
                    only a check, since coil shading makes the lean reference rough), in-plane pixel > --max-px mm (3),
                    axial slice > --max-thick mm, fewer than --min-slices uncut level slices
            check:  in-plane pixel > --check-px mm (2), less than --min-abd mm of abdomen
                    below the lungs, stored non-axial (reformatted), arms beside the trunk,
                    some level slices cut, fluid bright (gallbladder/liver > 2: T2-like), kidneys bright
                    (kidney/liver > --enh-kidney 1.6: contrast-enhanced; unenhanced CHAOS
                    in-phase reaches 1.36 where the liver is dark)
            ok:     none of the above.  The guess column says T1-like / T2-like / fat-sat? /
                    enhanced? from the same ratios; the pictures decide.

Outputs in --out: screen_volumes.csv (one row per volume; case ids only, no dates),
sheets/<prefix>_NNNN.png (three axial slices across the level with the trunk outline yellow,
arms magenta, edges the trunk runs along in red, and a coronal reformat with the level band), and
contact_NN.png (25 volumes a page, the middle level slice, for a first pass).
Columns added for the level: n_lung (slices with lung), abd_mm.

Needs numpy, scipy, nibabel, matplotlib.
"""
import argparse, csv, re, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mri_trunk import trunk_mask          # keep mri_trunk.py beside this script

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--root', required=True)
ap.add_argument('--out', default='runs/chaos_screen')
ap.add_argument('--meta', default='', help='labeled_data_meta_0000_0599.csv from the Zenodo record (scanner, site)')
ap.add_argument('--prefix', default='chaos', help='file name stem before _NNNN')
ap.add_argument('--min-id', type=int, default=None, help='lowest case number read (default 500 for amos, else 0)')
ap.add_argument('--cases', default='', help='comma list of case numbers (e.g. 507,541)')
ap.add_argument('--body-frac', type=float, default=0.08)
ap.add_argument('--edge-frac', type=float, default=0.02, help='share of an edge the body may run along before the slice counts as cut')
ap.add_argument('--min-fat', type=float, default=1.2, help='fat/liver below this: fat not bright')
ap.add_argument('--min-fat-lean', type=float, default=1.5, help='fat/lean (no liver label) below this: fat not bright')
ap.add_argument('--max-px', type=float, default=3.0, help='reject above this in-plane pixel of the axial slice (mm)')
ap.add_argument('--check-px', type=float, default=2.0, help='flag above this in-plane pixel (mm)')
ap.add_argument('--lung-min', type=float, default=300.0, help='smallest dark region (mm^2) counted as lung')
ap.add_argument('--lung-frac', type=float, default=0.4, help='lung = darker than this x the trunk median (or the body threshold if higher)')
ap.add_argument('--median', type=int, default=5, help='median filter size before the body threshold (1 = none)')
ap.add_argument('--min-abd', type=float, default=100.0, help='flag when less abdomen than this (mm) lies below the lungs')
ap.add_argument('--max-thick', type=float, default=10.0, help='largest acceptable axial slice spacing (mm)')
ap.add_argument('--min-slices', type=int, default=5)
ap.add_argument('--enh-kidney', type=float, default=1.6, help='kidney/liver above this: contrast-enhanced? (unenhanced CHAOS reaches 1.36)')
ap.add_argument('--arm-open', type=float, default=40.0, help='opening radius (mm) that removes the arms (narrower than twice this)')
ap.add_argument('--arm-rec', type=float, default=10.0, help='the opened trunk is grown back by this much (mm) inside the body mask')
ap.add_argument('--no-png', action='store_true')
a = ap.parse_args()
if a.min_id is None:
    a.min_id = 500 if a.prefix == 'amos' else 0

import nibabel as nib

root, out = Path(a.root).expanduser(), Path(a.out)
(out / 'sheets').mkdir(parents=True, exist_ok=True)
want = {int(x) for x in a.cases.split(',') if x.strip()}

LIVER, SPLEEN, GALL, AORTA, KIDNEYS = 6, 1, 4, 8, (2, 3)
SPLIT = {'tr': 'train', 'va': 'val', 'ts': 'test'}


def case_of(p):
    m = re.match(re.escape(a.prefix) + r'_(\d+)(_[A-Za-z0-9]+)*\.nii(\.gz)?$', p.name)
    return int(m.group(1)) if m else None


def split_of(d):
    s = d.name.lower()
    for k, v in SPLIT.items():
        if s.endswith(k):
            return v
    return ''


images, labels = {}, {}
for p in sorted(root.rglob(f'{a.prefix}_*.nii*')):
    c = case_of(p)
    if c is None or c < a.min_id or (want and c not in want):
        continue
    d = p.parent.name.lower()
    if d.startswith('image'):
        if c in images:
            print(f'{p.name}: a second image for case {c}, kept {images[c].name}')
            continue
        images[c] = p
    elif d.startswith('label') or d.startswith('mask'):
        labels[c] = p
if not images:
    raise SystemExit(f'no {a.prefix}_NNNN.nii[.gz] with id >= {a.min_id} under an image folder in {root}')

meta = {}
if a.meta:
    with open(a.meta, newline='', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            try:
                meta[int(r['amos_id'])] = (r.get("Manufacturer's Model Name", '').strip(), r.get('Site', '').strip())
            except (KeyError, ValueError):
                pass

print(f'{len(images)} MRI volumes, {sum(c in labels for c in images)} with labels')


def acq_plane(img):
    z = np.array(img.header.get_zooms()[:3], float)
    if z.max() / z.min() < 1.2:
        return '3D'
    code = nib.aff2axcodes(img.affine)[int(np.argmax(z))]
    return {'R': 'sagittal', 'L': 'sagittal', 'A': 'coronal', 'P': 'coronal', 'S': 'axial', 'I': 'axial'}[code]


def body_slices(v, thr, px):
    """per axial slice: trunk mask, arms (pieces of the body outside the trunk over 500 mm^2),
    arm share, and whether a notch cut was needed (mri_trunk.trunk_mask)"""
    T = np.zeros(v.shape, bool); A = np.zeros(v.shape, bool)
    arms = np.zeros(v.shape[2]); cuts = np.zeros(v.shape[2], int)
    st = ndi.generate_binary_structure(2, 1)
    for k in range(v.shape[2]):
        # median 5x5 first: speckle and ghosting outside the skin are sparse bright pixels that
        # the closing would otherwise weld into a fringe around the body (0025, 0029)
        sl = ndi.median_filter(v[:, :, k], size=a.median) if a.median > 1 else v[:, :, k]
        m = np.pad(sl > thr, 4)   # pad, or the closing erodes a body that reaches the edge
        m = ndi.binary_closing(m, st, iterations=2)[4:-4, 4:-4]
        m = ndi.binary_fill_holes(m)
        if not m.any():
            continue
        t, info = trunk_mask(m, px, img=v[:, :, k], lr_axis=0, open_mm=a.arm_open, rec_mm=a.arm_rec)
        rest = m & ~t
        cc, n = ndi.label(rest)
        if n:
            big = np.flatnonzero(np.bincount(cc.ravel())[1:] * px * px >= 500) + 1
            rest = np.isin(cc, big)
        T[:, :, k] = t; A[:, :, k] = rest
        arms[k] = rest.sum() / max(t.sum(), 1); cuts[k] = info['notch_cuts']
    return T, A, arms, cuts


def lung_run(v, T, thr, px):
    """per slice True where the lungs show: dark pieces >= --lung-min mm^2 in the posterior
    half of the trunk on both sides of the midline, followed down from the top of the stack"""
    nz = v.shape[2]; flag = np.zeros(nz, bool)
    for k in range(nz):
        t = T[:, :, k]
        if t.sum() < 100:
            continue
        # lungs are darker than every tissue but not always below the body threshold
        # (noisy volumes, 0030): dark = below the larger of that threshold and
        # --lung-frac x the trunk's median on this slice
        dthr = max(thr, a.lung_frac * np.median(v[:, :, k][t]))
        dark = ndi.binary_opening(t & (v[:, :, k] < dthr))
        cc, n = ndi.label(dark)
        if not n:
            continue
        xs, ys = np.nonzero(t)
        xc, ymid = np.median(xs), (ys.min() + ys.max()) / 2
        area = np.bincount(cc.ravel())[1:] * px * px
        sides = set()
        for i in np.flatnonzero(area >= a.lung_min) + 1:
            cx, cy = np.argwhere(cc == i).mean(0)
            if cy < ymid or area[i - 1] >= 10 * a.lung_min:   # RAS: low y = posterior; big lungs anywhere
                sides.add(cx > xc)
        flag[k] = len(sides) == 2
    lung = np.zeros(nz, bool)
    has = np.flatnonzero(T.any((0, 1)))
    if not len(has):
        return lung
    upper = has[has >= has.max() - 0.4 * (has.max() - has.min())]     # RAS: high z = superior
    start = [k for k in upper if flag[k]]
    if not start:                                          # no lung in the upper 40%: none
        return lung
    k, gap = max(start), 0                                 # the run starts at the highest flagged slice
    while k >= 0 and gap <= 1:
        if flag[k]:
            lung[k] = True; gap = 0
        else:
            gap += 1
        k -= 1
    lung[max(start):] = lung[max(start):] | flag[max(start):]   # anything flagged above it is lung too
    return lung


def edges_touched(m, frac):
    """edges the body runs along, in image terms of the RAS slice: x = patient left->right, y = posterior->anterior"""
    e = {'R': m[-1, :], 'L': m[0, :], 'A': m[:, -1], 'P': m[:, 0]}
    return [k for k, s in e.items() if s.mean() > frac]


rows = []
t0 = time.time()
for c in sorted(images):
    img = nib.load(str(images[c]))
    plane = acq_plane(img)
    can = nib.as_closest_canonical(img)
    v = np.asanyarray(can.dataobj).astype(np.float32)
    if v.ndim == 4:
        v = v[..., 0]
    zx, zy, zz = (float(x) for x in can.header.get_zooms()[:3])
    lab = None
    if c in labels:
        L = nib.as_closest_canonical(nib.load(str(labels[c])))
        lab = np.asanyarray(L.dataobj).astype(np.int16)
        if lab.shape != v.shape:
            print(f'amos_{c:04d}: label shape {lab.shape} != image {v.shape}, label ignored'); lab = None
    thr = a.body_frac * np.percentile(v, 99.5)
    B, Aslab, arms, notch = body_slices(v, thr, min(zx, zy))
    has = np.flatnonzero(B.any((0, 1)))
    lung = lung_run(v, B, thr, min(zx, zy))
    low = np.flatnonzero(lung).min() if lung.any() else (has.max() + 1 if len(has) else 0)
    abd = has[has < low]                                    # slices below the lungs
    if lab is not None and np.isin(lab, KIDNEYS).any():
        lev = np.flatnonzero(np.isin(lab, KIDNEYS).any((0, 1))); src = 'kidney'
    elif len(abd):
        lev = abd; src = 'below lungs'
    else:
        lev = np.array([], int); src = 'none'
    cut = {k: edges_touched(B[:, :, k], a.edge_frac) for k in lev}
    n_cut = sum(bool(e) for e in cut.values())
    cut_edges = sorted({x for e in cut.values() for x in e})
    # contrast
    med = lambda m: float(np.median(v[m])) if m is not None and m.any() else np.nan
    liver = med(lab == LIVER) if lab is not None else np.nan
    spleen = med(lab == SPLEEN) if lab is not None else np.nan
    gall = med(lab == GALL) if lab is not None else np.nan
    aorta = med(lab == AORTA) if lab is not None else np.nan
    kidney = med(np.isin(lab, KIDNEYS)) if lab is not None else np.nan
    rim_vals, body_vals = [], []
    for k in lev:
        b = B[:, :, k]
        if not b.any():
            continue
        d = ndi.distance_transform_edt(b, sampling=(zx, zy))
        rim = b & (d >= 2) & (d <= 12)
        if rim.any():
            rim_vals.append(v[:, :, k][rim])
        body_vals.append(v[:, :, k][b])
    fat = float(np.percentile(np.concatenate(rim_vals), 90)) if rim_vals else np.nan
    bv = np.concatenate(body_vals) if body_vals else np.array([np.nan])
    bmed = float(np.median(bv))
    lean = float(np.percentile(bv[bv > thr], 30)) if (bv > thr).any() else np.nan   # lean tissue even when half the trunk is fat
    use_liver = bool(np.isfinite(liver) and liver > 0)
    ref = liver if use_liver else lean
    min_fat = a.min_fat if use_liver else a.min_fat_lean
    ratio = lambda x: x / ref if np.isfinite(x) and np.isfinite(ref) and ref > 0 else np.nan
    fat_r, gall_r, spl_r, ao_r, kid_r = ratio(fat), ratio(gall), ratio(spleen), ratio(aorta), ratio(kidney)
    guess = []
    if np.isfinite(fat_r) and fat_r < min_fat:
        guess.append('fat-sat?')
    if np.isfinite(gall_r) and gall_r > 2:
        guess.append('T2-like')
    elif np.isfinite(gall_r) and gall_r < 0.8 and not guess:
        guess.append('T1-like')
    if np.isfinite(kid_r) and kid_r > a.enh_kidney:
        guess.append('enhanced?')
    rej, chk = [], []
    if np.isfinite(fat_r) and fat_r < min_fat:
        (rej if use_liver else chk).append(f'fat not bright ({fat_r:.2f})')   # without a liver label the lean reference is rough
    if max(zx, zy) > a.max_px:
        rej.append(f'in-plane {max(zx, zy):.2f} mm')
    elif max(zx, zy) > a.check_px:
        chk.append(f'in-plane {max(zx, zy):.2f} mm')
    if len(abd) * zz < a.min_abd:
        chk.append(f'{len(abd) * zz:.0f} mm below the lungs')
    if zz > a.max_thick:
        rej.append(f'slice {zz:.1f} mm')
    if len(lev) - n_cut < a.min_slices:
        rej.append(f'{len(lev) - n_cut} uncut level slices')
    if plane not in ('axial', '3D'):
        chk.append(f'stored {plane}')
    if len(lev) and np.median(arms[lev]) > 0.05:
        chk.append('arms')
    if n_cut:
        chk.append(f'{n_cut}/{len(lev)} level slices cut ({",".join(cut_edges)})')
    if 'T2-like' in guess:
        chk.append('fluid bright')
    if 'enhanced?' in guess:
        chk.append('kidneys bright')
    verdict = 'reject' if rej else 'check' if chk else 'ok'
    sc, site = meta.get(c, ('', ''))
    row = dict(case=f'{a.prefix}_{c:04d}', split=split_of(images[c].parent), scanner=sc, site=site,
               nx=v.shape[0], ny=v.shape[1], nz=v.shape[2], px_x=round(zx, 3), px_y=round(zy, 3), slice_mm=round(zz, 3),
               stored_plane=plane, labels=int(lab is not None), level_from=src, n_lung=int(lung.sum()), abd_mm=round(len(abd) * zz, 1),
               n_level=len(lev), n_level_cut=n_cut,
               cut_edges=''.join(cut_edges),
               arm_slices=int(Aslab[:, :, lev].any((0, 1)).sum()) if len(lev) else 0,
               arm_cut_slices=int((notch[lev] > 0).sum()) if len(lev) else 0, arms=round(float(np.median(arms[lev])) if len(lev) else 0, 3),
               liver=round(liver, 1), spleen=round(spleen, 1), gallbladder=round(gall, 1), aorta=round(aorta, 1), kidney=round(kidney, 1),
               fat_rim=round(fat, 1), body_median=round(bmed, 1), lean_p30=round(lean, 1), ratio_to=('liver' if use_liver else 'lean'),
               fat_ratio=round(fat_r, 3), gall_ratio=round(gall_r, 3), spleen_ratio=round(spl_r, 3), aorta_ratio=round(ao_r, 3), kidney_ratio=round(kid_r, 3),
               guess=' '.join(guess), verdict=verdict, reasons='; '.join(rej + chk))
    rows.append(row)
    print(f"{row['case']} {row['split']:5} {sc[:12]:12} {v.shape} {zx:.2f}x{zy:.2f}x{zz:.2f} {plane:8} "
          f"level {len(lev)} ({src}) cut {n_cut} arms on {row['arm_slices']}  fat {fat_r:.2f} gall {gall_r:.2f}  {verdict}: {row['reasons']}")

    if a.no_png:
        continue
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    hi = np.percentile(v[B], 99.5) if B.any() else v.max()
    disp = lambda s: np.clip(s.T[::-1, ::-1] / hi, 0, 1)   # anterior up, patient right on the viewer's left
    mdisp = lambda m: m.T[::-1, ::-1].astype(float)          # same orientation for masks, unscaled
    picks = [lev[int(f * (len(lev) - 1))] for f in (0.15, 0.5, 0.85)] if len(lev) else []
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    for i, k in enumerate(picks):
        ax[i].imshow(disp(v[:, :, k]), cmap='gray', aspect=zy / zx)
        ax[i].contour(mdisp(B[:, :, k]), [0.5], colors='yellow', linewidths=0.6)
        if Aslab[:, :, k].any():
            ax[i].contour(mdisp(Aslab[:, :, k]), [0.5], colors='magenta', linewidths=0.6)
        ny_, nx_ = v.shape[1], v.shape[0]
        for e in cut.get(k, []):   # draw the touched edge in display coordinates
            xy = {'R': ([0, 0], [0, ny_ - 1]), 'L': ([nx_ - 1, nx_ - 1], [0, ny_ - 1]),
                  'A': ([0, nx_ - 1], [0, 0]), 'P': ([0, nx_ - 1], [ny_ - 1, ny_ - 1])}[e]
            ax[i].plot(*xy, color='red', lw=3)
        ax[i].set_title(f'axial {k} of {v.shape[2]}', fontsize=9)
    j = v.shape[1] // 2
    if lab is not None and np.isin(lab, KIDNEYS).any():
        j = int(np.median(np.nonzero(np.isin(lab, KIDNEYS))[1]))
    cor = v[:, j, :]
    ax[3].imshow(np.clip(cor.T[::-1, ::-1] / hi, 0, 1), cmap='gray', aspect=zz / zx)
    if len(lev):
        for k in (lev.min(), lev.max()):
            ax[3].axhline(v.shape[2] - 1 - k, color='cyan', lw=1.6)
    if lung.any():
        ax[3].axhline(v.shape[2] - 1 - np.flatnonzero(lung).min(), color='magenta', lw=1.6)
    ax[3].set_title(f'coronal {j} (level band cyan, lowest lung slice magenta)', fontsize=9)
    for x in ax:
        x.axis('off')
    fig.suptitle(f"{row['case']} {row['split']} {sc} {site} | {zx:.2f}x{zy:.2f} mm, slice {zz:.2f} mm, stored {plane} | "
                 f"fat/{row['ratio_to']} {fat_r:.2f} gall {gall_r:.2f} kidney {kid_r:.2f} [{row['guess']}] | {verdict}: {row['reasons']}",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'sheets' / f"{row['case']}.png", dpi=80); plt.close(fig)
    row['_mid'] = (disp(v[:, :, lev[len(lev) // 2]]) if len(lev) else None, zy / zx)

with open(out / 'screen_volumes.csv', 'w', newline='') as fh:
    keys = [k for k in rows[0] if not k.startswith('_')]
    w = csv.DictWriter(fh, keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

if not a.no_png:
    import matplotlib.pyplot as plt
    for p0 in range(0, len(rows), 25):
        page = rows[p0:p0 + 25]
        fig, ax = plt.subplots(5, 5, figsize=(15, 15))
        for x in ax.ravel():
            x.axis('off')
        for x, r in zip(ax.ravel(), page):
            im, asp = r.get('_mid', (None, 1))
            if im is not None:
                x.imshow(im, cmap='gray', aspect=asp)
            x.set_title(f"{r['case'].rsplit('_', 1)[1]} {r['scanner'][:8]} {r['guess']} | {r['verdict']}",
                        fontsize=8, color={'ok': 'green', 'check': 'darkorange', 'reject': 'red'}[r['verdict']])
        fig.tight_layout(); fig.savefig(out / f'contact_{p0 // 25 + 1:02d}.png', dpi=70); plt.close(fig)

n = {k: sum(r['verdict'] == k for r in rows) for k in ('ok', 'check', 'reject')}
print(f"\n{len(rows)} volumes in {time.time() - t0:.0f} s: ok {n['ok']}, check {n['check']}, reject {n['reject']}")
from collections import Counter
print('guess:', dict(Counter(r['guess'] or '?' for r in rows)))
print('stored plane:', dict(Counter(r['stored_plane'] for r in rows)))
print(f'-> {out}/screen_volumes.csv' + ('' if a.no_png else f', {out}/sheets/, {out}/contact_*.png'))
