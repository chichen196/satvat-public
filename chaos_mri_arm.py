#!/usr/bin/env python3
"""chaos_mri_arm.py - the MRI arm on CHAOS T1-DUAL in-phase: threshold VAT against VIBESegmentator (Oct 2026).

For each case, on the abdominal slices (below the lungs; slices whose trunk runs into an
image edge are left out), three threshold-era splits and the model are measured on the
same trunk:

    meth   the corrected 2006 method (fatvol.est_fat_vol, Ridler-Calvard threshold, its own
           SAT/VAT labels), given the trunk as body mask so the arms never enter
    zone   the fat of the method's threshold split by the method's visceral zone
    wall   the same fat split by the wall-following rule (wallsep.py).  --wall-set mri
           (default) is the MRI setting: skin layer taken from the image (the non-fat
           contiguous with the outside, to --wall-skin-max 10 mm), closing 25 mm, no leak
           repair, and a second pass where the ring is open (cavity fat outside the zone):
           wall material below the upper three-class Otsu threshold of the trunk, admitted
           only deeper than the expected wall depth (--wall-high 1; the thin anterolateral
           wall of obese subjects lies above the fat threshold on 9 mm slices).  --wall-set
           ct is the AATTCT setting (fixed 3 mm skin, closing 9.4 mm, leak repair on),
           which fails on MRI - see wallsep.py for why.  Any --wall-* option overrides its
           preset.
    model  VIBESegmentator model 100: SAT = subcutaneous fat (65) inside the trunk minus the
           skin, VAT = inner fat (67) inside the trunk.  The model labels the dark band at
           the skin as SAT (10-12% of its SAT in the test cases, where a 2 mm rim holds
           only 1.5-3%), so the main SAT figure (--main-rim band) leaves out the same
           image-derived skin band the wall rule uses; fixed rims of 1, 2, 3 mm are also
           reported (--rims).

with and without N4 bias-field correction (SimpleITK, fitted inside the trunk mask) of the
image the threshold methods see (--n4 both|on|off; the model always gets the original).
Threshold VAT the model does not call VAT is broken down by what the model calls it
(its SAT, intestine, organ, muscle, bone, unlabelled); model VAT outside the threshold's
fat is reported as dim fat missed.

    cd satvat-public               # the repository folder, where fatvol imports
    python3 chaos_mri_arm.py --root /path/to/CHAOS --cases 1,30,33 --out mri_arm/try
    python3 chaos_mri_arm.py --root /path/to/CHAOS --out mri_arm/results

Inputs: <root>/mri_arm/inputs/chaos_NNNN.nii.gz and segs/chaos_NNNN_seg.nii.gz (vibe_pilot.py
run), <root>/labelsTr/chaos_NNNN.nii.gz where present (kidney slices marked).
Outputs in --out (relative to <root>): arm_slices.csv (per slice and variant, areas in pixels),
arm_cases.csv (per case and variant, ml and % differences, VAT breakdown), arm_summary.txt,
png/chaos_NNNN.png (middle abdominal slice: image | 2006 split | wall split | model split |
threshold VAT the model rejects, by tissue).  Case keys only.

Needs fatvol (segment.py with the Oct-2026 bdmask_in patch), wallsep.py, mri_trunk.py,
mri_body.py, numpy, scipy, nibabel, SimpleITK (pip install --user SimpleITK), matplotlib.
"""
import argparse, csv, re, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--root', required=True)
ap.add_argument('--arm', default='mri_arm', help='folder under --root holding inputs/ and segs/')
ap.add_argument('--out', default='mri_arm/results', help='output folder (relative to --root unless absolute)')
ap.add_argument('--cases', default='', help='comma list of case numbers')
ap.add_argument('--n4', default='both', choices=['both', 'on', 'off'])
ap.add_argument('--rims', default='band,1,2,3', help="skin left out of the model SAT: 'band' = the image-derived skin band "
                "(non-fat contiguous with the outside, to --wall-skin-max), or mm from the trunk edge")
ap.add_argument('--main-rim', default='band', help='which of --rims gives the main SAT figure')
ap.add_argument('--wall-set', default='mri', choices=['mri', 'ct'], help='wall-rule preset (see above)')
ap.add_argument('--wall-open', type=float, default=0.78)
ap.add_argument('--wall-close', type=float, default=None, help='closing radius mm (preset: mri 25, ct 9.4)')
ap.add_argument('--wall-open-post', type=float, default=3.1, help='posterior opening mm (0 = none)')
ap.add_argument('--wall-skin', type=float, default=3.0, help='fixed skin layer mm (floor)')
ap.add_argument('--wall-skin-max', type=float, default=None, help='skin band from the image, cap mm (preset: mri 10, ct none; 0 = none)')
ap.add_argument('--wall-leak', type=int, default=None, choices=[0, 1], help='leak repair (preset: mri 0, ct 1)')
ap.add_argument('--wall-high', type=int, default=None, choices=[0, 1], help='second pass with the brighter wall material where the ring is open (preset: mri 1, ct 0)')
ap.add_argument('--abd-up', type=int, default=0, help="fatvol orientation flag; 0 = anterior at row 0, as fed here")
ap.add_argument('--png', default='all', help="'all', 'none', or a comma list of case numbers")
ap.add_argument('--dpi', type=int, default=90)
a = ap.parse_args()

import nibabel as nib
from mri_body import body_trunk, lung_run, abdomen_slices, edge_cut
from wallsep import wall_zone, skin_band, otsu3

root = Path(a.root).expanduser()
arm = root / a.arm
out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
(out / 'png').mkdir(parents=True, exist_ok=True)
want = {int(x) for x in a.cases.split(',') if x.strip()}
RIMS = [x.strip() if x.strip() == 'band' else float(x) for x in a.rims.split(',') if x.strip()]
RIM0 = 'band' if a.main_rim == 'band' else float(a.main_rim)
if RIM0 not in RIMS:
    RIMS.insert(0, RIM0)
rim_name = lambda r: 'band' if r == 'band' else f'r{r:g}'
VARIANTS = {'both': ['raw', 'n4'], 'on': ['n4'], 'off': ['raw']}[a.n4]
PRESET = {'mri': dict(close_mm=25.0, close_steps_mm=(), skin_max_mm=10.0, leak=False),
          'ct': dict(close_mm=9.4, close_steps_mm=(15.0,), skin_max_mm=None, leak=True)}[a.wall_set]
WALL_HIGH = (a.wall_set == 'mri') if a.wall_high is None else bool(a.wall_high)
if a.wall_close is not None:
    PRESET['close_mm'] = a.wall_close; PRESET['close_steps_mm'] = ()
if a.wall_skin_max is not None:
    PRESET['skin_max_mm'] = a.wall_skin_max if a.wall_skin_max > 0 else None
if a.wall_leak is not None:
    PRESET['leak'] = bool(a.wall_leak)
WALL_KW = dict(open_mm=a.wall_open, skin_mm=a.wall_skin, anterior='up',
               open_post_mm=a.wall_open_post if a.wall_open_post > 0 else None, **PRESET)
SKIN_CAP = PRESET['skin_max_mm'] if PRESET['skin_max_mm'] is not None else 10.0   # for the 'band' rim

# VIBESegmentator model 100 (as in vibe_pilot.py)
SAT_L, VAT_L = 65, 67
GROUPS = {
    'model_sat': [SAT_L],
    'intestine': [6, 18, 19],
    'organ': [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22] + list(range(24, 42)),
    'muscle': list(range(53, 63)) + [66],
    'bone': [23] + list(range(42, 52)) + [52, 63, 64, 68, 69, 70, 71, 72],
    'none': [0],
}

try:
    from fatvol.segment import est_fat_vol, SUBCUTANEOUS, _ridler_calvard
except ImportError as e:
    raise SystemExit(f'fatvol does not import ({e}); run from the repository folder')


def to_grid(seg_img, ref_img):
    if seg_img.shape == ref_img.shape and np.allclose(seg_img.affine, ref_img.affine, atol=1e-3):
        return np.asarray(seg_img.dataobj).astype(np.int16)
    from nibabel.processing import resample_from_to
    return np.asarray(resample_from_to(seg_img, (ref_img.shape, ref_img.affine), order=0).dataobj).astype(np.int16)


def canon(arr, img):
    return np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(arr, img.affine)).dataobj)


def n4(v, mask, sp):
    import SimpleITK as sitk
    im = sitk.GetImageFromArray(np.transpose(np.maximum(v, 0) + 1.0, (2, 1, 0)).astype(np.float32))
    im.SetSpacing(tuple(float(x) for x in sp))
    mk = sitk.GetImageFromArray(np.transpose(mask.astype(np.uint8), (2, 1, 0)))
    mk.CopyInformation(im)
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations([50, 50, 30, 20])
    shrink = [2, 2, 1]
    f.Execute(sitk.Shrink(im, shrink), sitk.Shrink(mk, shrink))
    logb = f.GetLogBiasFieldAsImage(im)
    corr = sitk.GetArrayFromImage(im / sitk.Exp(logb))
    return np.transpose(corr, (2, 1, 0)).astype(np.float32) - 1.0


def s2(x):
    """[x, y] RAS slice -> rows anterior first, columns patient right first (as displayed)"""
    return x.T[::-1, ::-1]


def total_fat(r):
    for name in ('fatmask', 'tfatmask', 'totfatmask', 'tfmask', 'fat_mask', 'totalfat_mask', 'totalfat', 'tmask'):
        if hasattr(r, name):
            return getattr(r, name), name
    return None, None


_warned = set()


def run_meth(G, B, px, py, sp):
    """the 2006 method on a stack (nz, rows, cols) with the trunk as body mask; returns
    SAT, VAT, zone (visceral), total fat (method's threshold), each (nz, rows, cols) bool"""
    dy = dx = 0
    rr = np.flatnonzero(B.any(axis=(0, 2))); cc_ = np.flatnonzero(B.any(axis=(0, 1)))
    dy = int(round((G.shape[1] - 1) / 2 - (rr[0] + rr[-1]) / 2))
    dx = int(round((G.shape[2] - 1) / 2 - (cc_[0] + cc_[-1]) / 2))

    def shift(x, sy, sx):
        o = np.zeros_like(x)
        ny_, nx_ = x.shape[-2:]
        ys, yd = (slice(0, ny_ - sy), slice(sy, ny_)) if sy >= 0 else (slice(-sy, ny_), slice(0, ny_ + sy))
        xs, xd = (slice(0, nx_ - sx), slice(sx, nx_)) if sx >= 0 else (slice(-sx, nx_), slice(0, nx_ + sx))
        o[..., yd, xd] = x[..., ys, xs]
        return o
    if shift(B, dy, dx).sum() != B.sum():
        dy = dx = 0
    T = lambda x: np.ascontiguousarray(np.transpose(shift(x, dy, dx), (2, 1, 0)))
    U = lambda x: shift(np.transpose(x, (2, 1, 0)), -dy, -dx)
    r = est_fat_vol(T(G).astype(np.float32), px, py, sp, fat_loc=SUBCUTANEOUS, bdmask_in=T(B), abd_up=a.abd_up)
    MS, MV, Z = U(r.fmask) > 0, U(r.vfmask) > 0, U(r.zones) == 2
    tf, name = total_fat(r)
    if tf is None:
        if 'tf' not in _warned:
            fields = [k for k in vars(r) if isinstance(getattr(r, k), np.ndarray)]
            print(f'  NOTE: no total-fat mask among the result fields {fields}; using SAT | VAT | '
                  f'fat in neither from the zones (check the FatResult field names)', file=sys.stderr)
            _warned.add('tf')
        F = MS | MV
    else:
        F = U(tf) > 0
    return MS & B, MV & B, Z & B, F & B, getattr(r, 'is_abd_up', '')


roots = {}
for p in sorted((arm / 'inputs').glob('chaos_*.nii*')):
    m = re.match(r'chaos_(\d+)', p.name)
    if m and (not want or int(m.group(1)) in want):
        roots[int(m.group(1))] = p
if not roots:
    raise SystemExit(f'no inputs in {arm / "inputs"}')
png_cases = (set(roots) if a.png == 'all' else set() if a.png == 'none'
             else {int(x) for x in a.png.split(',') if x.strip()})

slice_rows, case_rows = [], []
t0 = time.time()
for cid, ip in roots.items():
    key = f'chaos_{cid:04d}'
    sp_ = arm / 'segs' / f'{key}_seg.nii.gz'
    if not sp_.exists():
        print(f'{key}: no segmentation'); continue
    img = nib.load(str(ip))
    v = np.asanyarray(nib.as_closest_canonical(img).dataobj).astype(np.float32)
    if v.ndim == 4:
        v = v[..., 0]
    zx, zy, zz = (float(x) for x in nib.as_closest_canonical(img).header.get_zooms()[:3])
    seg = canon(to_grid(nib.load(str(sp_)), img), img).astype(np.int16)
    lp = root / 'labelsTr' / f'{key}.nii.gz'
    lab = canon(np.asanyarray(nib.load(str(lp)).dataobj).astype(np.int16), img) if lp.exists() else None
    if lab is not None and lab.shape != v.shape:
        lab = None

    Tm, Am, notch, thr = body_trunk(v, min(zx, zy))
    lung = lung_run(v, Tm, thr, min(zx, zy))
    abd = abdomen_slices(Tm, lung)
    cut = [k for k in abd if edge_cut(Tm[:, :, k])]
    slab = np.array(sorted((k for k in abd if k not in cut), reverse=True))   # superior -> inferior
    if len(slab) < 3:
        print(f'{key}: {len(slab)} usable abdominal slices, skipped'); continue
    kid = set(np.flatnonzero(np.isin(lab, (2, 3)).any((0, 1)))) if lab is not None else set()

    B = np.stack([s2(Tm[:, :, k]) for k in slab])
    SEG = np.stack([s2(seg[:, :, k]) for k in slab])
    depth = np.stack([np.where(b, ndi.distance_transform_edt(b, sampling=(zy, zx)) - 0.5 * min(zx, zy), 0) for b in B])
    # the skin band from the original image: non-fat (below the slice's Ridler-Calvard
    # threshold inside the trunk) contiguous with the outside, to SKIN_CAP mm deep
    G0 = np.stack([s2(v[:, :, k]) for k in slab])
    BAND = np.zeros_like(B)
    for i in range(len(slab)):
        if B[i].any():
            thr_i = _ridler_calvard(G0[i] * B[i])
            BAND[i] = skin_band(B[i] & ~(G0[i] >= thr_i), B[i], (zx + zy) / 2, skin_max_mm=SKIN_CAP)
    MSAT = {rim: (SEG == SAT_L) & B & (~BAND if rim == 'band' else (depth > rim)) for rim in RIMS}
    MVAT = (SEG == VAT_L) & B
    msat_all = float(np.count_nonzero((SEG == SAT_L) & B))
    vox = zx * zy * zz / 1000.0
    ml = lambda m: float(np.count_nonzero(m)) * vox

    images = {'raw': v}
    if 'n4' in VARIANTS:
        t1 = time.time()
        try:
            images['n4'] = n4(v, Tm, (zx, zy, zz))
        except Exception as e:
            print(f'{key}: N4 failed ({type(e).__name__}: {e})')
        else:
            print(f'{key}: N4 {time.time() - t1:.0f} s')
    pic = {}
    for var in VARIANTS:
        if var not in images:
            continue
        G = np.stack([s2(images[var][:, :, k]) for k in slab])
        try:
            MS, MV, Z, F, abd_up = run_meth(G, B, zx, zy, zz)
        except Exception as e:
            print(f'{key} {var}: 2006 method FAILED {type(e).__name__}: {e}')
            case_rows.append(dict(case=key, variant=var, status=f'failed: {e}')); continue
        WZ = np.zeros_like(F)
        n_pass2 = 0
        for i in range(len(slab)):
            high = (B[i] & (G[i] < otsu3(G[i][B[i]])[1])) if (WALL_HIGH and B[i].any()) else None
            WZ[i], winfo = wall_zone(F[i], B[i], (zx + zy) / 2, nonfat=B[i] & ~F[i], nonfat_high=high, **WALL_KW)
            n_pass2 += int(winfo.get('pass', 1) >= 2)
        splits = dict(meth=(MS, MV), zone=(F & ~Z, F & Z), wall=(F & ~WZ, F & WZ))
        # misassignment of threshold fat against the model, by direction (the wall rule's own
        # error, separate from the threshold's): model SAT called VAT, model VAT called SAT
        msf, mvf = np.count_nonzero(MSAT[RIM0] & F), np.count_nonzero(MVAT & F)
        for i, k in enumerate(slab):
            row = dict(case=key, variant=var, z=int(k), kidney=int(k in kid), trunk_px=int(B[i].sum()), fat_px=int(F[i].sum()))
            for t_, (s_, v_) in splits.items():
                row[f'{t_}_sat'] = int(s_[i].sum()); row[f'{t_}_vat'] = int(v_[i].sum())
            for rim in RIMS:
                row[f'model_sat_{rim_name(rim)}'] = int(MSAT[rim][i].sum())
            row['model_vat'] = int(MVAT[i].sum())
            row['skin_band_px'] = int(BAND[i].sum())
            slice_rows.append(row)
        crow = dict(case=key, variant=var, status='ok', slices=len(slab), cut_left_out=len(cut), lung_slices=int(lung.sum()),
                    kidney_slices=len(kid & set(slab)), px=round(zx, 3), slice_mm=round(zz, 3), abd_up=abd_up,
                    wall_pass2_slices=n_pass2)
        msat, mvat = ml(MSAT[RIM0]), ml(MVAT)
        crow.update(model_sat_ml=round(msat, 1), model_vat_ml=round(mvat, 1), main_rim=rim_name(RIM0),
                    model_sat_in_skin_band_pct=round(100 * np.count_nonzero((SEG == SAT_L) & B & BAND) / max(msat_all, 1), 2))
        for rim in RIMS:
            crow[f'model_sat_{rim_name(rim)}_ml'] = round(ml(MSAT[rim]), 1)
        for t_, (s_, v_) in splits.items():
            ts, tv = ml(s_), ml(v_)
            crow.update({f'{t_}_sat_ml': round(ts, 1), f'{t_}_vat_ml': round(tv, 1),
                         f'{t_}_sat_diff_pct': round(100 * (ts - msat) / msat, 2) if msat else '',
                         f'{t_}_vat_diff_pct': round(100 * (tv - mvat) / mvat, 2) if mvat else '',
                         f'{t_}_vat_dice': round(2 * np.count_nonzero(v_ & MVAT) / max(np.count_nonzero(v_) + np.count_nonzero(MVAT), 1), 3),
                         f'{t_}_sat_to_vat_pct': round(100 * np.count_nonzero(v_ & MSAT[RIM0]) / max(msf, 1), 2),
                         f'{t_}_vat_to_sat_pct': round(100 * np.count_nonzero(s_ & MVAT) / max(mvf, 1), 2),
                         # the same two counts as a share of ALL threshold fat, the CT table's measure
                         f'{t_}_sat_to_vat_pctF': round(100 * np.count_nonzero(v_ & MSAT[RIM0]) / max(np.count_nonzero(F), 1), 3),
                         f'{t_}_vat_to_sat_pctF': round(100 * np.count_nonzero(s_ & MVAT) / max(np.count_nonzero(F), 1), 3),
                         f'{t_}_misassigned_pctF': round(100 * (np.count_nonzero(v_ & MSAT[RIM0]) + np.count_nonzero(s_ & MVAT)) / max(np.count_nonzero(F), 1), 3)})
        # what the model calls the 2006 method's VAT that it does not call VAT
        rej = MV & ~MVAT
        crow['meth_vat_rejected_ml'] = round(ml(rej), 1)
        crow['meth_vat_rejected_pct'] = round(100 * ml(rej) / ml(MV), 2) if ml(MV) else ''
        for g, codes in GROUPS.items():
            crow[f'rej_{g}_pct'] = round(100 * np.count_nonzero(rej & np.isin(SEG, codes)) / max(np.count_nonzero(rej), 1), 1)
        crow['model_vat_below_thr_ml'] = round(ml(MVAT & ~F), 1)
        crow['model_vat_below_thr_pct'] = round(100 * ml(MVAT & ~F) / mvat, 2) if mvat else ''
        case_rows.append(crow)
        print(f"{key} {var:3}: {len(slab)} slices ({len(cut)} cut left out)  VAT ml model {mvat:.0f} | meth {crow['meth_vat_ml']:.0f} "
              f"zone {crow['zone_vat_ml']:.0f} wall {crow['wall_vat_ml']:.0f}  rejected {crow['meth_vat_rejected_pct']}%  "
              f"SAT diff meth {crow['meth_sat_diff_pct']}% wall {crow['wall_sat_diff_pct']}%  "
              f"wall S->V {crow['wall_sat_to_vat_pct']}% V->S {crow['wall_vat_to_sat_pct']}%")
        pic[var] = (G, splits, rej)

    if cid not in png_cases or not pic:
        continue
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    var = 'n4' if 'n4' in pic else 'raw'
    G, splits, rej = pic[var]
    i = len(slab) // 2
    g = G[i]; hi = np.percentile(g[B[i]], 99.5) if B[i].any() else g.max()
    gray = np.dstack([np.clip(g / hi, 0, 1)] * 3)

    def paint(sat, vat):
        o = gray.copy()
        o[sat] = 0.35 * o[sat] + 0.65 * np.array((1, .55, 0))
        o[vat] = 0.35 * o[vat] + 0.65 * np.array((0, .8, 1))
        return o
    rejcol = {'model_sat': (1, .55, 0), 'intestine': (0.2, 0.9, 0.2), 'organ': (0.9, 0.2, 0.9),
              'muscle': (1, 0, 0), 'bone': (1, 1, 0), 'none': (1, 1, 1)}
    ro = gray.copy()
    for gname, codes in GROUPS.items():
        m = rej[i] & np.isin(SEG[i], codes)
        ro[m] = 0.3 * ro[m] + 0.7 * np.array(rejcol[gname])
    panels = [(gray, f'{key} z{slab[i]} ({var})'), (paint(*[x[i] for x in splits['meth']]), '2006 method'),
              (paint(*[x[i] for x in splits['wall']]), f'wall rule ({a.wall_set} setting)'),
              (paint(MSAT[RIM0][i], MVAT[i]), 'model (skin band off)' if RIM0 == 'band' else f'model (skin {RIM0:g} mm off)'),
              (ro, '2006 VAT not model VAT: SAT org, bowel grn, organ mag, muscle red, bone yel, none wht')]
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    for x, (im_, t_) in zip(ax, panels):
        x.imshow(im_, aspect=zy / zx); x.set_title(t_, fontsize=8); x.axis('off')
    fig.tight_layout(); fig.savefig(out / 'png' / f'{key}.png', dpi=a.dpi); plt.close(fig)

if slice_rows:
    with open(out / 'arm_slices.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(slice_rows[0])); w.writeheader(); w.writerows(slice_rows)
if case_rows:
    keys = []
    for r_ in case_rows:
        keys += [k for k in r_ if k not in keys]
    with open(out / 'arm_cases.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(case_rows)

lines = []
for var in VARIANTS:
    rs = [r_ for r_ in case_rows if r_.get('variant') == var and r_.get('status') == 'ok']
    if not rs:
        continue
    q = lambda col: np.array([float(r_[col]) for r_ in rs if r_.get(col) not in ('', None)])
    med = lambda x: f'{np.median(x):+.1f} ({np.percentile(x, 25):+.1f} to {np.percentile(x, 75):+.1f})' if len(x) else 'n/a'
    skin = 'skin band off' if RIM0 == 'band' else f'skin {RIM0:g} mm off'
    lines.append(f'== {var}: {len(rs)} cases; median (IQR) of % difference against the model (SAT {skin}; wall rule: {a.wall_set} setting) ==')
    for t_ in ('meth', 'zone', 'wall'):
        lines.append(f'  {t_:5} VAT {med(q(t_ + "_vat_diff_pct"))}   SAT {med(q(t_ + "_sat_diff_pct"))}   VAT Dice {np.median(q(t_ + "_vat_dice")):.3f}'
                     f'   fat misassigned: model SAT->VAT {np.median(q(t_ + "_sat_to_vat_pct")):.1f}%  model VAT->SAT {np.median(q(t_ + "_vat_to_sat_pct")):.1f}%'
                     f'   (of all threshold fat: {np.median(q(t_ + "_misassigned_pctF")):.2f}%)')
    lines.append(f'  model SAT lying in the skin band (median): {np.median(q("model_sat_in_skin_band_pct")):.1f}%'
                 f';  wall rule second pass used on {int(q("wall_pass2_slices").sum())} of {int(q("slices").sum())} slices in {int((q("wall_pass2_slices") > 0).sum())} cases')
    lines.append(f'  2006 VAT the model does not call VAT: {med(q("meth_vat_rejected_pct"))} % of it; by model label (median %): '
                 + ', '.join(f'{g} {np.median(q("rej_" + g + "_pct")):.0f}' for g in GROUPS))
    lines.append(f'  model VAT below the threshold (dim fat missed): {med(q("model_vat_below_thr_pct"))} % of model VAT')
txt = '\n'.join(lines)
print('\n' + txt + f'\n\n{len(case_rows)} case-variants in {time.time() - t0:.0f} s -> {out}')
(out / 'arm_summary.txt').write_text(txt + '\n')
