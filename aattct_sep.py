#!/usr/bin/env python3
"""aattct_sep.py - the 2006 method's SAT/VAT SEPARATION scored against AATTCT-IDS.

AATTCT-IDS images are windowed 8-bit JPEGs in which fat, air and bowel gas all sit
near grey 0, so the method's threshold cannot run on them.  This script tests only
the separation: every method gets the same fat mask F, and is scored on how it
splits F into SAT and VAT against the annotators.

    F      = body & grey <= --fat-hi & (SAT label | VAT label)    (fat pixels only;
             this drops the annotators' polygon overreach onto skin and fascia)
    seed   = F & grey <= --cfat-hi   (stands in for the 1.3x 'confident fat' mask
             that seeds the method's conditional dilation of SAT)
    body   = filled skin contour (grey > --body) united with the labels

Three splits are scored:
    meth : the 2006 method's own labels (pfatmask = SAT, vfatmask = VAT, rest = neither)
    zone : the 2006 method's visceral zone (zones == 2) applied to F; no 'neither'
    wall : the wall-following zone of wallsep.py applied to F (Oct 2026 replacement
           for the ellipse boundary); --overlay picks which split the PNGs show

Each subject's annotated slices (5 mm apart) are run as one volume in file-number
order, because the method carries its ellipse guess from slice to slice.

    aattct_sep.py --root /path/to/AATTCT-IDS --n 3 --png 6 --abd-up 0 --out runs/aattct_sep_up0
    aattct_sep.py --root ... --n 3 --png 6 --abd-up 1 --out runs/aattct_sep_up1
        (pilot: the orientation flag is a property of how the images are stored,
         so settle it once on a few subjects, then run all with that value)
    aattct_sep.py --root ... --n 0 --abd-up <0|1> --png 20 --out runs/aattct_sep

Wall-method settings (14 Oct 2026): --wall-open / --wall-skin for the whole body,
--wall-open-post / --wall-skin-post behind the body centre (the patient faces up in
these images), where the back's membranous layer and thick dermis otherwise survive
the opening and become "wall".  --no-meth skips the 2006 method (wall only, fast) and
--half odd|even keeps half the subjects, so settings can be chosen on one half and
reported on the other:

    aattct_sep.py --root ... --abd-up 0 --n 0 --half odd --no-meth --png 0 \
                  --wall-open 1.5 --wall-open-post 3.1 --out runs/wsweep_C

The wall debug picture (with --debug) shows the wall material before and after the
opening, the closed ring and the zone.

Outputs: sep_slices.csv, sep_subjects.csv, params.txt, a printed summary, overlay PNGs
(both SAT orange, both VAT cyan, annotators' VAT called SAT red, annotators' SAT
called VAT blue, method 'neither' magenta).  Pixel spacing is unknown (images were
cropped), so --px only scales the method's mm-based constants; results are in pixels.

Needs fatvol (with the Oct-2026 fatmask_in patch in segment.py), numpy, scipy,
pillow, matplotlib.  Run from the repository folder so `fatvol` imports.
"""
import argparse, csv, random, re, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from wallsep import wall_zone

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--n', type=int, default=3, help='random subjects (0 = all)')
ap.add_argument('--subjects', default='', help='comma list of patient_NNN, overrides --n')
ap.add_argument('--fat-hi', type=float, default=79, help='fat = grey <= this (p95 of SAT labels, aattct_probe)')
ap.add_argument('--cfat-hi', type=float, default=40, help='confident-fat seed = grey <= this')
ap.add_argument('--body', type=float, default=40)
ap.add_argument('--px', type=float, default=0.78, help='assumed pixel size (mm); unknown after cropping')
ap.add_argument('--spacing', type=float, default=5.0)
ap.add_argument('--abd-up', default='auto', choices=['auto', '0', '1'])
ap.add_argument('--png', type=int, default=6)
ap.add_argument('--no-center', action='store_true',
                help='do not move the body to the image centre (the method works in polar '
                     'coordinates about the IMAGE centre and assumes a centred body)')
ap.add_argument('--overlay', default='meth', choices=['meth', 'zone', 'wall'], help='which split the overlay PNGs show')
ap.add_argument('--wall-nonfat', default='grey', choices=['grey', 'labelled'],
                help="wall material for the wall method: 'grey' = body & grey > --fat-hi (all fat and gas excluded, "
                     "the realistic input); 'labelled' = body & ~F (unlabelled fat counts as wall)")
ap.add_argument('--wall-close', type=float, default=10.0, help='wall method: closing radius (mm)')
ap.add_argument('--wall-open', type=float, default=1.5,
                help='wall method: opening radius (mm), rounded to whole pixels; 0.78 with --wall-close 9.4 '
                     'reproduces runs made before 14 Oct 2026')
ap.add_argument('--wall-skin', type=float, default=3.0, help='wall method: skin layer excluded from the wall (mm)')
ap.add_argument('--wall-open-post', type=float, default=None,
                help='wall method: opening radius (mm) behind the body centre (default: same as --wall-open)')
ap.add_argument('--wall-skin-post', type=float, default=None,
                help='wall method: skin layer behind the body centre (mm) (default: same as --wall-skin)')
ap.add_argument('--wall-post-from', type=float, default=0.0,
                help='move the front/back dividing line backwards by this fraction of the body half-depth')
ap.add_argument('--no-meth', action='store_true', help='skip the 2006 method (meth/zone splits): wall only, fast')
ap.add_argument('--half', default='', choices=['', 'odd', 'even'], help='keep subjects with odd or even numbers only')
ap.add_argument('--no-wall', action='store_true', help='skip the wall method')
ap.add_argument('--debug', default='', help="comma list of patient_NNN:FILEnnn - write the method's intermediate masks for these slices")
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='aattct_sep')
a = ap.parse_args()
if a.no_meth and a.no_wall:
    sys.exit('--no-meth with --no-wall leaves nothing to score')
if a.no_meth:
    a.overlay = 'wall'
if not a.no_meth:
    from fatvol.segment import est_fat_vol, SUBCUTANEOUS
WALL_KW = dict(close_mm=a.wall_close, open_mm=a.wall_open, skin_mm=a.wall_skin)
if a.wall_open_post is not None or a.wall_skin_post is not None:
    WALL_KW.update(anterior='up', open_post_mm=a.wall_open_post, skin_post_mm=a.wall_skin_post,
                   post_from=a.wall_post_from)

root = Path(a.root).expanduser()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / 'params.txt').write_text(' '.join(sys.argv) + '\n' + '\n'.join(f'{k} = {v}' for k, v in sorted(vars(a).items())) + '\n')
abd_up = None if a.abd_up == 'auto' else int(a.abd_up)

def readlab(p):
    m = np.array(Image.open(p))
    return m.any(-1) if m.ndim == 3 else m > 0

def find(d, stem):
    for p in d.glob(stem + '.*'):
        return p
    return None

def num(p):
    d = re.findall(r'\d+', p.stem)
    return (int(d[-1]) if d else 0, p.stem)

subj = sorted(p for p in (root / 'Image' / 'Extracted').iterdir() if p.is_dir())
if a.subjects:
    want = set(s.strip() for s in a.subjects.split(','))
    subj = [p for p in subj if p.name in want]
elif a.n and a.n < len(subj):
    random.seed(a.seed); subj = sorted(random.sample(subj, a.n))
if a.half:
    subj = [p for p in subj if num(p)[0] % 2 == (1 if a.half == 'odd' else 0)]

def dice(x, y):
    s = x.sum() + y.sum()
    return 2.0 * (x & y).sum() / s if s else np.nan

rows, srows, pngs = [], [], 0
for sd in subj:
    t0 = time.time()
    G, S, V, ids = [], [], [], []
    for ip in sorted(sd.iterdir(), key=num):
        sp = find(root / 'Label' / 'Subcutaneous' / sd.name, ip.stem)
        vp = find(root / 'Label' / 'Visceral' / sd.name, ip.stem)
        if sp is None or vp is None:
            continue
        G.append(np.array(Image.open(ip).convert('L'))); S.append(readlab(sp)); V.append(readlab(vp)); ids.append(ip.stem)
    if not ids:
        continue
    G, S, V = np.stack(G), np.stack(S), np.stack(V)                     # (nz, rows, cols)
    B = np.stack([ndi.binary_fill_holes(ndi.binary_closing((g > a.body) | s | v, iterations=3))
                  for g, s, v in zip(G, S, V)])
    for i in range(len(B)):                                            # largest component (drops the table)
        cc, n = ndi.label(B[i])
        if n:
            B[i] = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
    F = B & (G <= a.fat_hi) & (S | V)
    C = F & (G <= a.cfat_hi)
    # centre the body: the bounding box of the body over all slices is moved to the
    # image centre for the method, and every output is moved back afterwards
    dy = dx = 0
    if not a.no_center:
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
        print(f'{sd.name}: centring would cut the body; not centred', file=sys.stderr); dy = dx = 0
    T = lambda x: np.ascontiguousarray(np.transpose(shift(x, dy, dx), (2, 1, 0)))  # (nz,rows,cols) -> (nx=cols, ny=rows, nz), centred
    U = lambda x: shift(np.transpose(x, (2, 1, 0)), -dy, -dx)                     # back
    dbg_want = [x.split(':', 1)[1] for x in a.debug.split(',') if x.startswith(sd.name + ':')]
    dbg = {} if (dbg_want and not a.no_meth) else None
    SPLITS = {}
    r = None
    if not a.no_meth:
        try:
            r = est_fat_vol(T(G).astype(np.float32), a.px, a.px, a.spacing, fat_loc=SUBCUTANEOUS,
                            fatmask_in=T(F), cfatmask_in=T(C), bdmask_in=T(B), abd_up=abd_up, debug=dbg)
        except Exception as e:
            print(f'{sd.name}: FAILED {type(e).__name__}: {e}', file=sys.stderr)
            srows.append(dict(subject=sd.name, n_slices=len(ids), status=f'failed: {e}'))
            continue
        MS = U(r.fmask) > 0; MV = U(r.vfmask) > 0; Z = U(r.zones) == 2
        MS &= F; MV &= F
        ZV = F & Z; ZS = F & ~Z
        SPLITS.update(meth=(MS, MV), zone=(ZS, ZV))
    # the wall-following separation (wallsep.py), slice by slice, in the original geometry
    WZ = np.zeros_like(F); winfo = []; wsteps = {}
    if not a.no_wall:
        for i in range(len(ids)):
            nf = (B[i] & (G[i] > a.fat_hi)) if a.wall_nonfat == 'grey' else (B[i] & ~F[i])
            if ids[i] in dbg_want:
                WZ[i], wi, wsteps[ids[i]] = wall_zone(F[i], B[i], a.px, nonfat=nf, return_steps=True, **WALL_KW)
            else:
                WZ[i], wi = wall_zone(F[i], B[i], a.px, nonfat=nf, **WALL_KW)
            winfo.append(wi)
        SPLITS['wall'] = (F & ~WZ, F & WZ)
    RS, RV = F & S, F & V
    for i, sid in enumerate(ids):
        row = dict(subject=sd.name, slice=sid, F_px=int(F[i].sum()),
                   ref_sat=int(RS[i].sum()), ref_vat=int(RV[i].sum()))
        if winfo:
            row.update(wall_close_mm=winfo[i]['close_mm'], wall_closed=int(winfo[i]['closed']),
                       wall_leak_bins=winfo[i]['leak_bins'], wall_leak_px=winfo[i]['leak_moved_px'])
        for tag, (msa, mva) in SPLITS.items():
            ms, mv = msa[i], mva[i]
            row.update({f'{tag}_sat': int(ms.sum()), f'{tag}_vat': int(mv.sum()),
                        f'{tag}_none': int((F[i] & ~ms & ~mv).sum()),
                        f'{tag}_vat2sat': int((RV[i] & ms).sum()), f'{tag}_sat2vat': int((RS[i] & mv).sum()),
                        f'{tag}_dice_sat': round(dice(ms, RS[i]), 4), f'{tag}_dice_vat': round(dice(mv, RV[i]), 4)})
        rows.append(row)
        if pngs < a.png:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            gray = np.dstack([G[i] / 255.0] * 3); ov = gray.copy()
            OS, OV = SPLITS[a.overlay][0][i], SPLITS[a.overlay][1][i]
            for m, c in ((RS[i] & OS, (1, .55, 0)), (RV[i] & OV, (0, .8, 1)),
                         (RV[i] & OS, (1, 0, 0)), (RS[i] & OV, (0, 0, 1)),
                         (F[i] & ~OS & ~OV, (1, 0, 1))):
                ov[m] = .35 * gray[m] + .65 * np.array(c)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
            ax[0].imshow(gray); ax[1].imshow(ov)
            for x in ax: x.axis('off')
            rv = row['ref_vat'] or 1
            ax[0].set_title(f'{sd.name} {sid}' + (f'  IsAbdUp={r.is_abd_up}' if r is not None else ''), fontsize=9)
            ov_v = row[f'{a.overlay}_vat']
            ax[1].set_title(f"VAT ref {row['ref_vat']} {a.overlay} {ov_v} ({(ov_v-rv)/rv*100:+.1f}%)  "
                            f"red: ref VAT->SAT  blue: ref SAT->VAT  magenta: neither", fontsize=8)
            fig.tight_layout(); fig.savefig(out / f'{sd.name}_{sid}.png', dpi=90); plt.close(fig)
            pngs += 1
    for sid, st in wsteps.items():
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        i = ids.index(sid); wi = winfo[i]
        gray = np.dstack([G[i] / 255.0] * 3)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        p1 = gray.copy()
        p1[st['N0'] & ~st['N']] = (1, .85, 0)                        # removed by the opening
        p1[st['N']] = (.2, .9, .2)                                    # kept as wall material
        ax[0].imshow(p1)
        ax[0].set_title('green: wall material after skin layer + opening  yellow: removed by the opening', fontsize=8)
        ax[1].imshow(gray)
        ax[1].contour(st['W'].astype(float), [.5], colors='lime', linewidths=.7)
        ax[1].contour(WZ[i].astype(float), [.5], colors='white', linewidths=1)
        if 'anterior' in WALL_KW:
            from wallsep import _posterior
            ax[1].contour(_posterior(B[i], 'up', a.wall_post_from).astype(float), [.5], colors='yellow',
                          linewidths=.6, linestyles='dashed')
        ax[1].set_title(f"lime: closed ring (close {wi['close_mm']:g} mm)  white: wall zone  "
                        f"yellow dash: posterior part  leak bins {wi['leak_bins']}", fontsize=8)
        ov = gray.copy(); RSi, RVi = F[i] & S[i], F[i] & V[i]; OS, OV = F[i] & ~WZ[i], F[i] & WZ[i]
        for m, c in ((RSi & OS, (1, .55, 0)), (RVi & OV, (0, .8, 1)), (RVi & OS, (1, 0, 0)), (RSi & OV, (0, 0, 1))):
            ov[m] = .35 * gray[m] + .65 * np.array(c)
        ax[2].imshow(ov)
        ax[2].set_title(f'{sd.name} {sid}  wall split  red: ref VAT->SAT  blue: ref SAT->VAT', fontsize=8)
        for x in ax: x.axis('off')
        fig.tight_layout(); fig.savefig(out / f'debug_wall_{sd.name}_{sid}.png', dpi=90); plt.close(fig)
        print(f'debug -> {out}/debug_wall_{sd.name}_{sid}.png')
    if dbg:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        D = {k: U(v) for k, v in dbg.items() if isinstance(v, np.ndarray) and v.ndim == 3}
        fr = shift(dbg['front'].T, -dy, -dx)                           # (rows, cols), original position
        for sid in dbg_want:
            if sid not in ids:
                print(f'debug: {sd.name}:{sid} not found', file=sys.stderr); continue
            i = ids.index(sid)
            gray = np.dstack([G[i] / 255.0] * 3)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
            for x in ax: x.imshow(gray); x.axis('off')
            ov = gray.copy(); ov[RV[i] & MS[i]] = (1, 0, 0)
            ov[fr & B[i]] = .75 * ov[fr & B[i]] + .25 * np.array((1, 1, 0))
            ax[0].imshow(ov); ax[0].set_title(f"red: ref VAT->SAT; yellow tint: 'front' sector (IsAbdUp={dbg['is_abd_up']})", fontsize=8)
            nf = gray.copy(); nf[D['rvnonfatmask'][i] > 0] = (.2, .9, .2)
            ax[1].imshow(nf); ax[1].set_title('green: rvnonfatmask (non-fat the visceral edge is fitted to)', fontsize=8)
            for k, c in (('pfmedgemask', 'orange'), ('rvnfmedgemask', 'lime'), ('rvisceromask', 'cyan'), ('crvisceromask', 'magenta')):
                ax[1 if k == 'rvnfmedgemask' else 2].contour(D[k][i].astype(float), [.5], colors=c, linewidths=1)
            ax[2].imshow(ov * 0 + gray); ax[2].contour((RV[i] & MS[i]).astype(float), [.5], colors='red', linewidths=.8)
            if not a.no_wall:
                ax[2].contour(WZ[i].astype(float), [.5], colors='white', linewidths=.8)
            ax[2].set_title('orange: SAT outer edge  lime(mid): fitted visceral edge  cyan: visceral zone  magenta: core  white: wall method', fontsize=8)
            fig.tight_layout(); fig.savefig(out / f'debug_{sd.name}_{sid}.png', dpi=90); plt.close(fig)
            print(f'debug -> {out}/debug_{sd.name}_{sid}.png')
    sub = [x for x in rows if x['subject'] == sd.name]
    tot = lambda k: sum(x[k] for x in sub)
    srows.append(dict(subject=sd.name, n_slices=len(ids), status='ok',
                      is_abd_up=r.is_abd_up if r is not None else '', dy=dy, dx=dx,
                      notes=len(r.notes) if r is not None else '', ref_sat=tot('ref_sat'), ref_vat=tot('ref_vat'),
                      **{f'{t}_{c}': tot(f'{t}_{c}') for t in SPLITS for c in ('sat', 'vat', 'none', 'vat2sat', 'sat2vat')}))
    print(f'{sd.name}: {len(ids)} slices' + (f', IsAbdUp={r.is_abd_up}, shift (dy,dx)=({dy},{dx})' if r is not None else '')
          + f', {time.time()-t0:.0f}s', flush=True)

if not rows:
    sys.exit('no slices processed')
for name, data in (('sep_slices.csv', rows), ('sep_subjects.csv', srows)):
    keys = list(dict.fromkeys(k for d in data for k in d))
    with open(out / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(data)

R = {k: np.array([x[k] for x in rows], float) for k in rows[0] if k not in ('subject', 'slice')}
ok = R['ref_vat'] > 0
sub = [s for s in srows if s.get('status') == 'ok']
print(f'\n{len(rows)} slices, {len(sub)} subjects ok, {len(srows)-len(sub)} failed'
      + ('' if a.no_meth else f'; IsAbdUp per subject: {sorted(set(s["is_abd_up"] for s in sub))}'))
if 'anterior' in WALL_KW or a.wall_open != 1.5 or a.wall_skin != 3.0 or a.wall_close != 10.0:
    print('wall settings: ' + ', '.join(f'{k}={v}' for k, v in WALL_KW.items()))
for t in SPLITS:
    e = (R[f'{t}_vat'][ok] - R['ref_vat'][ok]) / R['ref_vat'][ok] * 100
    es = (R[f'{t}_sat'] - R['ref_sat']) / np.maximum(R['ref_sat'], 1) * 100
    sv = np.array([s[f'{t}_vat'] for s in sub], float); rv = np.array([s['ref_vat'] for s in sub], float)
    ss = np.array([s[f'{t}_sat'] for s in sub], float); rs = np.array([s['ref_sat'] for s in sub], float)
    print(f'[{t}] per slice  VAT %err median {np.median(e):+.1f}  %RMSE {np.sqrt(np.mean(e**2)):.1f}   '
          f'SAT %err median {np.median(es):+.1f}  %RMSE {np.sqrt(np.mean(es**2)):.1f}')
    print(f'[{t}] Dice median  SAT {np.nanmedian(R[f"{t}_dice_sat"]):.3f}  VAT {np.nanmedian(R[f"{t}_dice_vat"]):.3f}')
    print(f'[{t}] of all F: ref VAT called SAT {R[f"{t}_vat2sat"].sum()/R["F_px"].sum():.2%}, '
          f'ref SAT called VAT {R[f"{t}_sat2vat"].sum()/R["F_px"].sum():.2%}, neither {R[f"{t}_none"].sum()/R["F_px"].sum():.2%}')
    if len(sub) > 1:
        ev = (sv - rv) / rv * 100
        print(f'[{t}] per subject (summed slices)  VAT %RMSE {np.sqrt(np.mean(ev**2)):.1f}  '
              f'SAT %RMSE {np.sqrt(np.mean(((ss-rs)/rs*100)**2)):.1f}  r(VAT) {np.corrcoef(sv, rv)[0,1]:.3f}')
        print(f'[{t}] per subject  VAT %err median {np.median(ev):+.1f}  subjects off by >5%: {int((abs(ev) > 5).sum())}  '
              f'>10%: {int((abs(ev) > 10).sum())}  of {len(sub)}')
print(f'-> {out}/sep_slices.csv, sep_subjects.csv, {pngs} PNGs')
