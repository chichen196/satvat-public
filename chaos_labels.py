#!/usr/bin/env python3
"""chaos_labels.py - CHAOS MR training labels (PNG per slice) -> NIfTI on the image grid (Oct 2026).

CHAOS gives the T1-DUAL labels as one PNG per slice in <subject>/T1DUAL/Ground, named like
the in-phase DICOMs (IMG-0027-00002.png <-> DICOM_anon/InPhase/IMG-0027-00002.dcm).  This
script places every label pixel on the grid of the NIfTI that dcm2niix made from the same
InPhase folder (imagesTr/chaos_NNNN[_e2].nii.gz), through each DICOM's own position and
orientation, so slice order, flips and dcm2niix's reorientation cannot misplace it.

    chaos_labels.py --root /path/to/CHAOS
        reads  <root>/Train_Sets/MR/<id>/T1DUAL/{Ground,DICOM_anon/InPhase}
               <root>/imagesTr/chaos_NNNN*.nii.gz
        writes <root>/labelsTr/chaos_NNNN.nii.gz        (AMOS codes, so the screen reads them:
                                                         1 spleen, 2 right kidney, 3 left kidney, 6 liver)
               <root>/labelsTr/check/chaos_NNNN.png     (image | label overlay, 3 slices)
               <root>/labelsTr/chaos_labels.csv         (voxels per organ, placement checks)

CHAOS grey levels (the challenge's own ranges): liver 55-70, right kidney 110-135, left kidney
175-200, spleen 240-255.  Right and left are the patient's.

Checks per subject: every PNG has a DICOM and every DICOM a PNG; every label pixel lands on
an image voxel within 0.1 voxel of a grid point; the liver label sits on homogeneous tissue
(coefficient of variation of the image inside it, printed beside the same measure for the
label flipped up-down and left-right: the true placement should be the lowest).  The PNGs
are taken as stored (row = DICOM row); if the flip check disagrees for a subject, --flip
ud|lr|both overrides it for that run and the picture shows the result.

Needs numpy, nibabel, pydicom, pillow, matplotlib.
"""
import argparse, csv, re, sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--root', required=True)
ap.add_argument('--subjects', default='', help='comma list of CHAOS ids (default: all training subjects)')
ap.add_argument('--flip', default='', choices=['', 'ud', 'lr', 'both'], help='flip the PNGs before placing them (only if the check says so)')
ap.add_argument('--no-png', action='store_true')
a = ap.parse_args()

import nibabel as nib
import pydicom
from PIL import Image

root = Path(a.root).expanduser()
mr = root / 'Train_Sets' / 'MR'
if not mr.is_dir():
    hits = sorted(root.rglob('Train_Sets/MR'))
    if not hits:
        raise SystemExit(f'no Train_Sets/MR under {root}')
    mr = hits[0]
out = root / 'labelsTr'
(out / 'check').mkdir(parents=True, exist_ok=True)
want = {int(x) for x in a.subjects.split(',') if x.strip()}

RANGES = [(55, 70, 6), (110, 135, 2), (175, 200, 3), (240, 255, 1)]   # grey range -> AMOS code
NAMES = {6: 'liver', 2: 'kidney_R', 3: 'kidney_L', 1: 'spleen'}


def png_codes(p):
    g = np.array(Image.open(p).convert('L'))
    lab = np.zeros(g.shape, np.int16)
    for lo, hi, code in RANGES:
        lab[(g >= lo) & (g <= hi)] = code
    return lab


def flipped(lab, how):
    if how in ('ud', 'both'):
        lab = lab[::-1]
    if how in ('lr', 'both'):
        lab = lab[:, ::-1]
    return lab


def world_lps_to_ras(xyz):
    return xyz * np.array([-1, -1, 1.0])


def place(dcms, pngs, img, how):
    """label volume on img's grid; returns (vol, max fractional offset, n pixels placed, n outside)"""
    inv = np.linalg.inv(img.affine)
    vol = np.zeros(img.shape[:3], np.int16)
    worst, placed, outside = 0.0, 0, 0
    for d, p in zip(dcms, pngs):
        ds = pydicom.dcmread(str(d), stop_before_pixels=True)
        ipp = np.array(ds.ImagePositionPatient, float)
        iop = np.array(ds.ImageOrientationPatient, float)
        dr, dc = (float(x) for x in ds.PixelSpacing)          # row spacing, column spacing
        lab = flipped(png_codes(p), how)
        if lab.shape != (int(ds.Rows), int(ds.Columns)):
            raise SystemExit(f'{p.name}: PNG {lab.shape} vs DICOM {(ds.Rows, ds.Columns)}')
        r, c = np.nonzero(lab)
        if not len(r):
            continue
        # DICOM: X (first triple) = direction of increasing column, Y = increasing row
        xyz = ipp + np.outer(c * dc, iop[:3]) + np.outer(r * dr, iop[3:])
        ijk = (inv @ np.c_[world_lps_to_ras(xyz), np.ones(len(r))].T)[:3].T
        rk = np.rint(ijk)
        worst = max(worst, float(np.abs(ijk - rk).max()))
        rk = rk.astype(int)
        ok = np.all((rk >= 0) & (rk < np.array(vol.shape)), axis=1)
        outside += int((~ok).sum()); placed += int(ok.sum())
        vol[rk[ok, 0], rk[ok, 1], rk[ok, 2]] = lab[r[ok], c[ok]]
    return vol, worst, placed, outside


def cv_inside(img_data, vol, code):
    m = vol == code
    if m.sum() < 50:
        return np.nan
    x = img_data[m]
    return float(x.std() / max(x.mean(), 1e-6))


images = {}
for p in sorted((root / 'imagesTr').glob('chaos_*.nii*')):
    m = re.match(r'chaos_(\d+)', p.name)
    if m:
        images.setdefault(int(m.group(1)), p)

rows = []
subs = sorted(int(d.name) for d in mr.iterdir() if d.is_dir() and d.name.isdigit())
for sid in subs:
    if want and sid not in want:
        continue
    t1 = mr / str(sid) / 'T1DUAL'
    gdir, ddir = t1 / 'Ground', t1 / 'DICOM_anon' / 'InPhase'
    if not gdir.is_dir() or not ddir.is_dir():
        print(f'{sid}: no T1DUAL/Ground or InPhase folder'); continue
    if sid not in images:
        print(f'{sid}: no imagesTr/chaos_{sid:04d}*.nii.gz (run dcm2niix first)'); continue
    dmap = {p.stem: p for p in ddir.glob('*.dcm')}
    pmap = {p.stem: p for p in gdir.glob('*.png')}
    common = sorted(set(dmap) & set(pmap))
    miss_png, miss_dcm = sorted(set(dmap) - set(pmap)), sorted(set(pmap) - set(dmap))
    img = nib.load(str(images[sid]))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    dcms = [dmap[s] for s in common]; pngs = [pmap[s] for s in common]
    cvs = {}
    for how in ('', 'ud', 'lr', 'both'):
        v, _, _, _ = place(dcms, pngs, img, how)
        cvs[how or 'none'] = cv_inside(data, v, 6)
    vol, worst, placed, outside = place(dcms, pngs, img, a.flip)
    used = a.flip or 'none'
    best = min(cvs, key=lambda k: cvs[k] if np.isfinite(cvs[k]) else 9)
    nib.save(nib.Nifti1Image(vol, img.affine), str(out / f'chaos_{sid:04d}.nii.gz'))
    row = dict(case=f'chaos_{sid:04d}', slices=len(common), png_without_dcm=len(miss_dcm), dcm_without_png=len(miss_png),
               voxels_placed=placed, voxels_outside=outside, max_offset_vox=round(worst, 3), flip_used=used,
               liver_cv_none=round(cvs['none'], 3), liver_cv_ud=round(cvs['ud'], 3), liver_cv_lr=round(cvs['lr'], 3),
               liver_cv_both=round(cvs['both'], 3), lowest_cv=best,
               **{NAMES[c] + '_vox': int((vol == c).sum()) for c in NAMES})
    rows.append(row)
    flag = '' if best == used else f'  <-- the liver sits best with flip {best}; look at the picture'
    print(f"{row['case']}: {len(common)} slices, {placed} label px placed, {outside} outside, offset {worst:.3f} vox, "
          f"liver CV none {cvs['none']:.3f} ud {cvs['ud']:.3f} lr {cvs['lr']:.3f} both {cvs['both']:.3f}{flag}")
    if miss_png or miss_dcm:
        print(f'   unmatched: {len(miss_png)} DICOM without PNG, {len(miss_dcm)} PNG without DICOM')

    if a.no_png:
        continue
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    can = nib.as_closest_canonical(img)
    cd = np.asanyarray(can.dataobj).astype(np.float32)
    if cd.ndim == 4:
        cd = cd[..., 0]
    cl = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(vol, img.affine)).dataobj)
    ks = np.flatnonzero(cl.any((0, 1)))
    if not len(ks):
        continue
    picks = [ks[int(f * (len(ks) - 1))] for f in (0.2, 0.5, 0.8)]
    hi = np.percentile(cd, 99.5)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    cols = {6: (0.9, 0.6, 0.1), 2: (0.1, 0.8, 0.2), 3: (0.1, 0.5, 1.0), 1: (0.9, 0.1, 0.6)}
    for j, k in enumerate(picks):
        g = np.clip(cd[:, :, k].T[::-1, ::-1] / hi, 0, 1)    # anterior up, patient right on the viewer's left
        L = cl[:, :, k].T[::-1, ::-1]
        ov = np.dstack([g] * 3)
        for c, rgb in cols.items():
            ov[L == c] = 0.5 * ov[L == c] + 0.5 * np.array(rgb)
        ax[0, j].imshow(g, cmap='gray'); ax[1, j].imshow(ov)
        ax[0, j].set_title(f'slice {k}', fontsize=9)
        for x in ax[:, j]:
            x.axis('off')
    fig.suptitle(f"{row['case']}  flip {used}  (liver orange, right kidney green, left kidney blue, spleen pink;"
                 f" patient's right on the viewer's left)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'check' / f"{row['case']}.png", dpi=80); plt.close(fig)

if rows:
    with open(out / 'chaos_labels.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r['case'] for r in rows if r['lowest_cv'] != r['flip_used']]
    print(f"\n{len(rows)} subjects -> {out}/  (labels, check/*.png, chaos_labels.csv)")
    print('flip check disagrees for: ' + (', '.join(bad) if bad else 'none'))
