#!/usr/bin/env python3
"""port_corrections.py - Table 2: what each correction to the 2006 method changes (Oct 2026).

The corrected method (the fatvol defaults) is run with one correction undone at a time:

  corrected       Ridler-Calvard threshold on each slice's own outline; only converged,
                  in-field ellipse fits accepted; fit jump test in both passes; orientation
                  from the image header
  idl_rule        the .pro's threshold loop under IDL's integer-histogram bin width
                  (threshold_rule='idl'; estfatvolmriv2.pro lines 293-336)
  slice0_mask     every slice's threshold taken inside the first slice's body outline (the
                  2003/2008 threshold program, line 176: images[*,*,j]*bdmask, 2-D x 3-D)
  no_visc_jump    no fit jump test in the visceral pass (the .pro as written)
  fit_fov         ellipse fits accepted converged or not, if inside the field of view
  fit_any         whatever the optimiser returns, no seed check (the port before 17 Sep 2026)
  port_prefix     fit_any + no_visc_jump (the port as it first ran)
  abd_auto        orientation from the .pro's non-fat test instead of the header
  pro_as_written  idl_rule + no_visc_jump + abd_auto: estfatvolmriv2.pro as written, with
                  IDL AMOEBA's handling of non-converged fits (the seed stands)

Two parts:

  phantom  parametric synthetic abdomens with known SAT and VAT: SAT thickness 6/18/32 mm,
           visceral fat 10/35% of the cavity, outline growing or shrinking 15% over 10
           slices (so slice 0's outline differs from the rest), with and without a 30%
           central coil shading, both orientations; 48 phantoms, no data needed
  chaos    the CHAOS in-phase cases, original images, scored against VIBESegmentator
           exactly as chaos_mri_arm.py does (same trunk, abdominal slices, model SAT minus
           the image-derived skin band, model VAT = inner fat)

    cd satvat-public                # the repository folder
    python3 port_corrections.py phantom --out runs/table2_phantom
    python3 port_corrections.py chaos --root /path/to/CHAOS --cases 1,30,33 --out mri_arm/table2_try
    python3 port_corrections.py chaos --root /path/to/CHAOS --out mri_arm/table2 --jobs 2

Time: with the 28 Oct idlcompat.py (row-run morphology for large disks) the method takes a
few seconds per case, so nine variants x 40 cases is about 10-20 min with --jobs 1.
Without it the runaway-fit variants can ask scipy for 10-30 GB per process: install the
whole 28 Oct fatvol bundle before running.

Outputs in --out (chaos: relative to --root unless absolute; case keys only):
  corr_cases.csv    one row per case (or phantom) and variant: volumes, change against
                    'corrected', thresholds, fit counts, orientation test, and for chaos the
                    agreement with the model (VAT difference, Dice, misassignment)
  corr_slices.csv   per slice and variant: threshold and fat / SAT / VAT pixels
  corr_summary.txt  per variant: median (IQR) changes, cases affected, failures
"""
import argparse, csv, re, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

VARIANTS = {
    'corrected':      dict(kw={}, abd='header'),
    'idl_rule':       dict(kw=dict(threshold_rule='idl'), abd='header'),
    'slice0_mask':    dict(kw=dict(thresh_mask_slice='first'), abd='header'),
    'no_visc_jump':   dict(kw=dict(visceral_jump_test=False), abd='header'),
    'fit_fov':        dict(kw=dict(fit_rule='fov'), abd='header'),
    'fit_any':        dict(kw=dict(fit_rule='any'), abd='header'),
    'port_prefix':    dict(kw=dict(fit_rule='any', visceral_jump_test=False), abd='header'),
    'abd_auto':       dict(kw={}, abd='auto'),
    'pro_as_written': dict(kw=dict(threshold_rule='idl', visceral_jump_test=False), abd='auto'),
}

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('part', choices=['phantom', 'chaos'])
ap.add_argument('--out', required=True)
ap.add_argument('--variants', default=','.join(VARIANTS), help='comma list (corrected is always run)')
ap.add_argument('--jobs', type=int, default=1, help='variants run in parallel (processes); each needs ~1 CPU and <1 GB')
ap.add_argument('--slice0', default='first', choices=['first', 'last'],
                help="slice whose outline slice0_mask uses: 'first' = slice 0 of the stack as the method "
                     "sees it (chaos: the most superior abdominal slice; phantom: k = 0)")
# chaos
ap.add_argument('--root', help='CHAOS root (chaos part)')
ap.add_argument('--arm', default='mri_arm', help='folder under --root holding inputs/ and segs/')
ap.add_argument('--cases', default='', help='comma list of case numbers (chaos part)')
ap.add_argument('--abd-up', type=int, default=0, help="orientation from the header, as chaos_mri_arm.py feeds it (0)")
# phantom
ap.add_argument('--phantom-size', type=int, default=256)
ap.add_argument('--phantom-slices', type=int, default=10)
ap.add_argument('--quick', action='store_true', help='phantom: 8 phantoms instead of 48 (a smoke test)')
a = ap.parse_args()

want_var = ['corrected'] + [v.strip() for v in a.variants.split(',') if v.strip() and v.strip() != 'corrected']
bad = [v for v in want_var if v not in VARIANTS]
if bad:
    sys.exit(f'unknown variants {bad}; choose from {list(VARIANTS)}')

try:
    from fatvol.segment import est_fat_vol, SUBCUTANEOUS, _ridler_calvard
    import fatvol.ellipse as E
    E.FIT_STATS; E.reset_fit_stats
    import inspect
    import fatvol.idlcompat as _ic
    if not hasattr(_ic, '_row_morph'):
        raise ImportError('this fatvol/idlcompat.py has no row-run morphology (install the 28 Oct 2026 bundle; '
                          'the old one can ask scipy for 10-30 GB on the runaway-fit variants)')
    if 'thresh_mask_slice' not in inspect.signature(est_fat_vol).parameters:
        raise ImportError('this fatvol has no thresh_mask_slice / fit_rule switches (install the 28 Oct 2026 '
                          'segment.py, ellipse.py and run_fatvol.py)')
except (ImportError, AttributeError) as e:
    raise SystemExit(f'fatvol does not import with the Table 2 switches ({e}); run from the repository folder')


def resolve_kw(kw, nz):
    kw = dict(kw)
    if kw.get('thresh_mask_slice') == 'first':
        kw['thresh_mask_slice'] = 0 if a.slice0 == 'first' else nz - 1
    return kw


def run_method(vol, B, px, py, sp, variant, abd_header):
    """vol, B in the method's (nx, ny, nz) order.  Returns a dict of masks and diagnostics,
    or raises."""
    spec = VARIANTS[variant]
    kw = resolve_kw(spec['kw'], vol.shape[2])
    abd = None if spec['abd'] == 'auto' else abd_header
    E.reset_fit_stats()
    dbg = {}
    t0 = time.time()
    r = est_fat_vol(vol.astype(np.float32), px, py, sp, fat_loc=SUBCUTANEOUS, bdmask_in=B,
                    abd_up=abd, debug=dbg, **kw)
    halves = dbg.get('abd_halves', (np.nan, np.nan, -1))
    return dict(F=r.fatmask > 0, S=r.fmask > 0, V=r.vfmask > 0, thr=np.asarray(r.thresholds, float),
                abd=r.is_abd_up, abd_detected=halves[2], halves=halves[:2], notes=list(r.notes),
                fit=dict(E.FIT_STATS), sec=time.time() - t0)


def _job(args):
    vol, B, px, py, sp, var, abd_header = args
    try:
        return var, run_method(vol, B, px, py, sp, var, abd_header)
    except Exception as e:                       # reported per variant, the others still run
        return var, e


def run_all(vol, B, px, py, sp, abd_header):
    """All wanted variants on one volume, in parallel when --jobs > 1 (fork, so nothing re-imports)."""
    args = [(vol, B, px, py, sp, var, abd_header) for var in want_var]
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(min(a.jobs, len(args))) as pool:
            res = dict(pool.map(_job, args, chunksize=1))
    else:
        res = dict(map(_job, args))
    return res


def diag(res):
    n = res['notes']
    l, u = res['halves']
    return dict(poor_sub_fits=sum('poor subcutaneous' in x for x in n),
                poor_visc_fits=sum('poor visceral' in x for x in n),
                fit_calls=res['fit']['calls'], fit_converged=res['fit']['converged'],
                fit_accepted=res['fit']['accepted'], seed_resets=res['fit']['seed_reset'],
                abd_used=res['abd'], abd_detected=res['abd_detected'],
                abd_margin=round((u - l) / (u + l), 4) if np.isfinite(l) and (u + l) > 0 else '')


def pct(x, ref):
    return round(100.0 * (x - ref) / ref, 3) if ref else ''


def write_csv(path, rows):
    keys = []
    for r_ in rows:
        keys += [k for k in r_ if k not in keys]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------------------
# phantoms
# ----------------------------------------------------------------------------------------
def make_abdomen(nx=256, nz=10, px=1.5, sp=7.0, sat_mm=18.0, vat_frac=0.25, taper=0.0,
                 bias=0.0, abdomen_up=False, seed=0, wall_mm=9.0):
    """Synthetic T1 abdomen in the method's (nx, ny, nz) order with truth 1 = SAT, 2 = VAT.

    Body: ellipse 0.72 x 0.52 of the field, scaled by 1 + taper * k/(nz-1) along z.  SAT ring
    sat_mm thick (1.35x at the flanks, 0.65x front and back), muscle wall wall_mm, cavity of
    mid-grey organs and bowel with visceral fat blobs filling vat_frac of it (anterior-
    weighted), paraspinal muscle and a vertebra behind.  abdomen_up follows io.make_phantom's
    convention (posterior at small j <-> IsAbdUp = 1).  Partial volume by a 0.8 px blur,
    central coil shading 1 - bias*(1 - rho^2), Rician-like noise (sd 12)."""
    rng = np.random.default_rng(seed)
    ny = nx
    i = np.arange(nx)[:, None].astype(float)
    j = np.arange(ny)[None, :].astype(float)
    X = (i - nx / 2.0) * np.ones((1, ny))
    Y = (j - ny / 2.0) * np.ones((nx, 1))
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    post = -1.0 if abdomen_up else 1.0            # posterior direction along j
    vol = np.zeros((nx, ny, nz)); truth = np.zeros((nx, ny, nz), np.uint8); body = np.zeros((nx, ny, nz), bool)
    tex = ndi.gaussian_filter(rng.normal(0, 1, (nx, ny)), 6)
    tex = tex / (np.abs(tex).max() + 1e-9) * 60.0
    for k in range(nz):
        s = 1.0 + taper * k / max(nz - 1, 1)
        A, Bq = 0.36 * nx * s, 0.26 * ny * s
        Rout = A * Bq / np.sqrt((Bq * np.cos(th)) ** 2 + (A * np.sin(th)) ** 2)
        t = sat_mm / px * (1.0 + 0.35 * np.cos(2 * th))
        Rsat = Rout - t
        Rwall = Rsat - wall_mm / px
        bd = r <= Rout
        sat = bd & (r > Rsat)
        wall = (r <= Rsat) & (r > Rwall)
        cav = r <= Rwall
        img = np.zeros((nx, ny))
        img[sat] = 900.0
        img[wall] = 320.0
        img[cav] = 400.0 + tex[cav]
        # paraspinal muscles and a vertebra behind the cavity's centre
        bw = Bq - (sat_mm + wall_mm) / px
        para = cav & ((X / (0.42 * A)) ** 2 + ((Y - post * 0.58 * bw) / (0.30 * bw)) ** 2 <= 1.0)
        img[para] = 330.0
        vert = np.hypot(X, Y - post * 0.42 * bw) <= 0.055 * nx
        img[vert & cav] = 450.0
        cortex = vert & ~(np.hypot(X, Y - post * 0.42 * bw) <= 0.055 * nx - 2)
        img[cortex & cav] = 150.0
        # visceral fat blobs, anterior-weighted, until vat_frac of the free cavity is fat
        free = cav & ~para & ~vert
        target = vat_frac * free.sum()
        vis = np.zeros((nx, ny), bool)
        ys, xs = np.nonzero(free)
        ant = -post * (Y[ys, xs] / max(bw, 1))          # +1 front, -1 back
        w = np.clip(0.55 + 0.45 * ant, 0.05, None); w /= w.sum()
        for _ in range(4000):
            if vis.sum() >= target:
                break
            q = rng.choice(len(ys), p=w)
            rad = rng.uniform(4, 12) / px
            vis |= (np.hypot(X - X[ys[q], xs[q]], Y - Y[ys[q], xs[q]]) <= rad) & free
        img[vis] = 880.0
        lab = np.zeros((nx, ny), np.uint8); lab[sat] = 1; lab[vis] = 2
        img = ndi.gaussian_filter(img, 0.8)
        if bias:
            rho = np.clip(r / Rout, 0, 1)
            img = img * np.where(bd, 1.0 - bias * (1.0 - rho ** 2), 1.0)
        img = np.abs(img + rng.normal(0, 12, img.shape))
        vol[:, :, k] = img; truth[:, :, k] = lab; body[:, :, k] = bd
    return vol, truth, body, dict(px=px, sp=sp, abd_up=1 if abdomen_up else 0)


def run_phantoms():
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = []
    for sat in (6.0, 18.0, 32.0):
        for vf in (0.10, 0.35):
            for taper in (-0.15, 0.15):
                for bias in (0.0, 0.3):
                    for up in (False, True):
                        grid.append(dict(sat_mm=sat, vat_frac=vf, taper=taper, bias=bias, abdomen_up=up))
    if a.quick:
        grid = grid[::6]
    rows, srows = [], []
    t0 = time.time()
    for n, g in enumerate(grid):
        key = (f"ph{n:02d}_sat{g['sat_mm']:g}_vf{g['vat_frac']:g}_tp{g['taper']:+g}_b{g['bias']:g}_"
               f"{'up' if g['abdomen_up'] else 'dn'}")
        vol, truth, body, m = make_abdomen(nx=a.phantom_size, nz=a.phantom_slices, seed=n, **g)
        vox = m['px'] * m['px'] * m['sp'] / 1000.0
        TS, TV = truth == 1, truth == 2
        base = None
        allres = run_all(vol, body.astype(np.uint8), m['px'], m['px'], m['sp'], m['abd_up'])
        for var in want_var:
            res = allres[var]
            if isinstance(res, Exception):
                e = res
                rows.append(dict(case=key, variant=var, status=f'failed: {type(e).__name__}: {e}'[:200], **g))
                print(f'{key} {var}: FAILED {type(e).__name__}: {e}'); continue
            F, S, V = res['F'], res['S'], res['V']
            row = dict(case=key, variant=var, status='ok', **g,
                       truth_sat_ml=round(TS.sum() * vox, 2), truth_vat_ml=round(TV.sum() * vox, 2),
                       thr_median=round(float(np.nanmedian(res['thr'])), 2),
                       total_ml=round(F.sum() * vox, 2), sat_ml=round(S.sum() * vox, 2), vat_ml=round(V.sum() * vox, 2),
                       vat_tms_ml=round((F.sum() - S.sum()) * vox, 2))
            row.update(vat_err_pct=pct(row['vat_ml'], row['truth_vat_ml']), sat_err_pct=pct(row['sat_ml'], row['truth_sat_ml']),
                       vat_tms_err_pct=pct(row['vat_tms_ml'], row['truth_vat_ml']),
                       vat_to_sat_pct=round(100 * (TV & S).sum() / max(TV.sum(), 1), 3),
                       sat_to_vat_pct=round(100 * (TS & V).sum() / max(TS.sum(), 1), 3),
                       sec=round(res['sec'], 1), **diag(res))
            if var == 'corrected':
                base = row
            if base is not None:
                row.update(d_thr=round(row['thr_median'] - base['thr_median'], 2),
                           d_total_pct=pct(row['total_ml'], base['total_ml']),
                           d_sat_pct=pct(row['sat_ml'], base['sat_ml']), d_vat_pct=pct(row['vat_ml'], base['vat_ml']))
            rows.append(row)
            for z in range(vol.shape[2]):
                srows.append(dict(case=key, variant=var, z=z, thr=round(float(res['thr'][z]), 2),
                                  fat_px=int(F[:, :, z].sum()), sat_px=int(S[:, :, z].sum()), vat_px=int(V[:, :, z].sum())))
        print(f'{key}: done ({time.time() - t0:.0f} s)', flush=True)
    write_csv(out / 'corr_cases.csv', rows)
    write_csv(out / 'corr_slices.csv', srows)
    summarise(rows, out, phantom=True)


# ----------------------------------------------------------------------------------------
# CHAOS
# ----------------------------------------------------------------------------------------
def run_chaos():
    import nibabel as nib
    from mri_body import body_trunk, lung_run, abdomen_slices, edge_cut
    from wallsep import skin_band
    if not a.root:
        sys.exit('--root is required for the chaos part')
    root = Path(a.root).expanduser()
    arm = root / a.arm
    out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
    out.mkdir(parents=True, exist_ok=True)
    want = {int(x) for x in a.cases.split(',') if x.strip()}
    SAT_L, VAT_L = 65, 67

    def to_grid(seg_img, ref_img):
        if seg_img.shape == ref_img.shape and np.allclose(seg_img.affine, ref_img.affine, atol=1e-3):
            return np.asarray(seg_img.dataobj).astype(np.int16)
        from nibabel.processing import resample_from_to
        return np.asarray(resample_from_to(seg_img, (ref_img.shape, ref_img.affine), order=0).dataobj).astype(np.int16)

    def canon(arr, img):
        return np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(arr, img.affine)).dataobj)

    s2 = lambda x: x.T[::-1, ::-1]       # [x, y] RAS slice -> rows anterior first, as chaos_mri_arm.py

    def shift(x, sy, sx):
        o = np.zeros_like(x)
        ny_, nx_ = x.shape[-2:]
        ys, yd = (slice(0, ny_ - sy), slice(sy, ny_)) if sy >= 0 else (slice(-sy, ny_), slice(0, ny_ + sy))
        xs, xd = (slice(0, nx_ - sx), slice(sx, nx_)) if sx >= 0 else (slice(-sx, nx_), slice(0, nx_ + sx))
        o[..., yd, xd] = x[..., ys, xs]
        return o

    inputs = {}
    for p in sorted((arm / 'inputs').glob('chaos_*.nii*')):
        mm = re.match(r'chaos_(\d+)', p.name)
        if mm and (not want or int(mm.group(1)) in want):
            inputs[int(mm.group(1))] = p
    if not inputs:
        sys.exit(f'no inputs in {arm / "inputs"}')
    (out / 'params.txt').write_text(' '.join(sys.argv) + '\n')
    rows, srows = [], []
    t_all = time.time()
    for cid, ip in inputs.items():
        key = f'chaos_{cid:04d}'
        img = nib.load(str(ip))
        cimg = nib.as_closest_canonical(img)
        v = np.asanyarray(cimg.dataobj).astype(np.float32)
        if v.ndim == 4:
            v = v[..., 0]
        zx, zy, zz = (float(x) for x in cimg.header.get_zooms()[:3])
        sp_ = arm / 'segs' / f'{key}_seg.nii.gz'
        seg = canon(to_grid(nib.load(str(sp_)), img), img).astype(np.int16) if sp_.exists() else None
        Tm, Am, notch, thr = body_trunk(v, min(zx, zy))
        lung = lung_run(v, Tm, thr, min(zx, zy))
        abd = abdomen_slices(Tm, lung)
        cut = [k for k in abd if edge_cut(Tm[:, :, k])]
        slab = np.array(sorted((k for k in abd if k not in cut), reverse=True))   # superior -> inferior
        if len(slab) < 3:
            print(f'{key}: {len(slab)} usable abdominal slices, skipped'); continue
        B = np.stack([s2(Tm[:, :, k]) for k in slab])
        G = np.stack([s2(v[:, :, k]) for k in slab])
        vox = zx * zy * zz / 1000.0
        if seg is not None:
            SEG = np.stack([s2(seg[:, :, k]) for k in slab])
            BAND = np.zeros_like(B)
            for i in range(len(slab)):
                if B[i].any():
                    t_i = _ridler_calvard(G[i] * B[i])
                    BAND[i] = skin_band(B[i] & ~(G[i] >= t_i), B[i], (zx + zy) / 2, skin_max_mm=10.0)
            MSAT = (SEG == SAT_L) & B & ~BAND
            MVAT = (SEG == VAT_L) & B
        rr = np.flatnonzero(B.any(axis=(0, 2))); cc = np.flatnonzero(B.any(axis=(0, 1)))
        dy = int(round((G.shape[1] - 1) / 2 - (rr[0] + rr[-1]) / 2))
        dx = int(round((G.shape[2] - 1) / 2 - (cc[0] + cc[-1]) / 2))
        if shift(B, dy, dx).sum() != B.sum():
            dy = dx = 0
        T = lambda x: np.ascontiguousarray(np.transpose(shift(x, dy, dx), (2, 1, 0)))
        U = lambda x: shift(np.transpose(x, (2, 1, 0)), -dy, -dx)
        base = None
        allres = run_all(T(G), T(B).astype(np.uint8), zx, zy, zz, a.abd_up)
        for var in want_var:
            res = allres[var]
            if isinstance(res, Exception):
                e = res
                rows.append(dict(case=key, variant=var, status=f'failed: {type(e).__name__}: {e}'[:200], slices=len(slab)))
                print(f'{key} {var}: FAILED {type(e).__name__}: {e}', flush=True); continue
            F, S, V = U(res['F']) & B, U(res['S']) & B, U(res['V']) & B
            row = dict(case=key, variant=var, status='ok', slices=len(slab), px=round(zx, 3), slice_mm=round(zz, 3),
                       thr_median=round(float(np.nanmedian(res['thr'])), 2),
                       total_ml=round(F.sum() * vox, 1), sat_ml=round(S.sum() * vox, 1), vat_ml=round(V.sum() * vox, 1),
                       vat_tms_ml=round((F.sum() - S.sum()) * vox, 1), sec=round(res['sec'], 1), **diag(res))
            if seg is not None:
                msat, mvat = MSAT.sum() * vox, MVAT.sum() * vox
                msf, mvf = np.count_nonzero(MSAT & F), np.count_nonzero(MVAT & F)
                s2v = 100 * np.count_nonzero(V & MSAT) / max(msf, 1)
                v2s = 100 * np.count_nonzero(S & MVAT) / max(mvf, 1)
                row.update(model_sat_ml=round(msat, 1), model_vat_ml=round(mvat, 1),
                           vat_diff_model_pct=pct(row['vat_ml'], mvat), sat_diff_model_pct=pct(row['sat_ml'], msat),
                           vat_dice=round(2 * np.count_nonzero(V & MVAT) / max(V.sum() + MVAT.sum(), 1), 3),
                           sat_to_vat_pct=round(s2v, 2), vat_to_sat_pct=round(v2s, 2), mis_sum_pct=round(s2v + v2s, 2),
                           mis_pctF=round(100 * (np.count_nonzero(V & MSAT) + np.count_nonzero(S & MVAT)) / max(F.sum(), 1), 3),
                           rejected_pct=round(100 * np.count_nonzero(V & ~MVAT) / max(V.sum(), 1), 2),
                           model_vat_below_thr_pct=round(100 * np.count_nonzero(MVAT & ~F) / max(MVAT.sum(), 1), 2))
            if var == 'corrected':
                base = dict(row, thr=res['thr'])
            if base is not None:
                row.update(d_thr=round(row['thr_median'] - base['thr_median'], 2),
                           d_thr_pct=round(float(np.nanmedian(100 * (res['thr'] - base['thr']) / base['thr'])), 2),
                           d_total_pct=pct(row['total_ml'], base['total_ml']),
                           d_sat_pct=pct(row['sat_ml'], base['sat_ml']), d_vat_pct=pct(row['vat_ml'], base['vat_ml']))
                if 'mis_sum_pct' in row and 'mis_sum_pct' in base:
                    row['d_mis_sum'] = round(row['mis_sum_pct'] - base['mis_sum_pct'], 2)
            rows.append(row)
            for i, k in enumerate(slab):
                srows.append(dict(case=key, variant=var, z=int(k), slice_in_stack=i, thr=round(float(res['thr'][i]), 2),
                                  trunk_px=int(B[i].sum()), fat_px=int(F[i].sum()), sat_px=int(S[i].sum()), vat_px=int(V[i].sum())))
            print(f"{key} {var:14}: thr {row['thr_median']:7.1f}  VAT {row['vat_ml']:7.1f} ml"
                  + (f"  ({row['d_vat_pct']:+.1f}% vs corrected)" if var != 'corrected' and row.get('d_vat_pct') != '' else '')
                  + (f"  mis {row['mis_sum_pct']:.1f}%" if 'mis_sum_pct' in row else '')
                  + f"  fits {row['fit_converged']}/{row['fit_calls']} conv  abd {row['abd_used']} (det {row['abd_detected']})"
                  + f"  {row['sec']:.0f} s", flush=True)
        write_csv(out / 'corr_cases.csv', rows)          # rewritten after every case
        write_csv(out / 'corr_slices.csv', srows)
    print(f'{len(inputs)} cases in {time.time() - t_all:.0f} s')
    summarise(rows, out, phantom=False)


# ----------------------------------------------------------------------------------------
def summarise(rows, out, phantom):
    lines = []
    med = lambda x: (f'{np.median(x):+.2f} ({np.percentile(x, 25):+.2f} to {np.percentile(x, 75):+.2f})'
                     if len(x) else 'n/a')
    num = lambda rs, k: np.array([float(r_[k]) for r_ in rs if r_.get(k) not in ('', None)])
    for var in want_var:
        rs = [r_ for r_ in rows if r_['variant'] == var]
        ok = [r_ for r_ in rs if r_.get('status') == 'ok']
        if not rs:
            continue
        lines.append(f'== {var}: {len(ok)} ok, {len(rs) - len(ok)} failed ==')
        if var != 'corrected':
            d = num(ok, 'd_vat_pct')
            lines.append(f'  change vs corrected, median (IQR) %: VAT {med(d)}   SAT {med(num(ok, "d_sat_pct"))}   '
                         f'total {med(num(ok, "d_total_pct"))}   threshold {med(num(ok, "d_thr"))} grey levels'
                         + ('' if phantom else f' ({med(num(ok, "d_thr_pct"))}%)'))
            if len(d):
                lines.append(f'  cases with VAT changed: >0.1% {int((abs(d) > 0.1).sum())}, >1% {int((abs(d) > 1).sum())}, '
                             f'>5% {int((abs(d) > 5).sum())} of {len(d)}; max |change| {np.max(abs(d)):.1f}%')
        if phantom:
            lines.append(f'  against truth, median (range) %: VAT {np.median(num(ok, "vat_err_pct")):+.1f} '
                         f'({np.min(num(ok, "vat_err_pct")):+.1f} to {np.max(num(ok, "vat_err_pct")):+.1f})   '
                         f'SAT {np.median(num(ok, "sat_err_pct")):+.1f}   '
                         f'truth VAT called SAT {np.median(num(ok, "vat_to_sat_pct")):.1f}%   '
                         f'truth SAT called VAT {np.median(num(ok, "sat_to_vat_pct")):.1f}%')
        elif ok and 'mis_sum_pct' in ok[0]:
            lines.append(f'  against the model, median (IQR): VAT difference {med(num(ok, "vat_diff_model_pct"))}%   '
                         f'misassignment sum {med(num(ok, "mis_sum_pct"))}%   VAT Dice {np.median(num(ok, "vat_dice")):.3f}   '
                         f'rejected VAT {med(num(ok, "rejected_pct"))}%')
        fc, fv = num(ok, 'fit_calls').sum(), num(ok, 'fit_converged').sum()
        lines.append(f'  ellipse fits: {int(fv)}/{int(fc)} converged ({100 * fv / max(fc, 1):.1f}%), '
                     f'{int(num(ok, "fit_accepted").sum())} accepted, {int(num(ok, "seed_resets").sum())} seed resets; '
                     f'jump-test rejections: subcutaneous {int(num(ok, "poor_sub_fits").sum())}, '
                     f'visceral {int(num(ok, "poor_visc_fits").sum())}')
        det = [r_ for r_ in ok if r_.get('abd_detected') not in ('', None, -1)]
        if det:
            hdr = [int(r_['abd_detected']) for r_ in det]
            if phantom:
                wrong = [r_ for r_ in det if int(r_['abd_detected']) != (1 if r_['abdomen_up'] else 0)]
            else:
                wrong = [r_ for r_ in det if int(r_['abd_detected']) != a.abd_up]
            lines.append(f'  orientation test reversed on {len(wrong)} of {len(det)}'
                         + (f": {', '.join(r_['case'] for r_ in wrong[:12])}" if wrong and not phantom else ''))
    txt = '\n'.join(lines)
    print('\n' + txt)
    (out / 'corr_summary.txt').write_text(txt + '\n')
    print(f'-> {out}/corr_cases.csv, corr_slices.csv, corr_summary.txt')


if a.part == 'phantom':
    run_phantoms()
else:
    run_chaos()
