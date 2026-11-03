#!/usr/bin/env python3
"""overlay_figs.py - Figures 2 and 5 of the manuscript: the overlay panels (Oct 2026).

  fig2  CT (AATCT-IDS): the same fat mask F split by the 2006 method, the wall rule
        (setting B) and the U-Net (f2m), against the annotators, on chosen slices.
        Columns: image | 2006 method | wall rule | U-Net.
  fig5  MRI (CHAOS in-phase, original images): why the wall rule needed an MRI setting.
        Rows by mechanism, columns: image | wall rule, CT setting | MRI setting, first pass |
        MRI setting, final | VIBESegmentator.
          skin   the low-signal skin band acts as wall and the ring closes at the skin
                 (default case 30; slice with the most model SAT called VAT by the CT setting)
          leak   the CT setting's leak repair fires on a thick flank pad (default case 1;
                 slice where the repair moved the most fat)
          pass2  a thin anterolateral wall above the fat threshold leaves the ring open; the
                 second pass closes it (default case 33; slice where the second pass removed
                 the most model VAT called SAT)

Colours (chosen to stay apart under colour-vision deficiency): agreed SAT pale orange,
agreed VAT pale blue, reference VAT called SAT red, reference SAT called VAT violet, fat
in neither compartment aqua (2006 method only).  In fig5 the reference is the model and
only threshold fat that the model labels SAT or VAT is coloured; the white line is the
wall rule's zone.

    cd satvat-public              # the repository folder
    python3 overlay_figs.py fig2 --root /path/to/AATTCT-IDS --unet-run runs/unet_f2m --out figs
    python3 overlay_figs.py fig2 ... --slices patient_074:FILE200,patient_138:FILE126
    python3 overlay_figs.py fig2 ... --candidates runs/aattct_sep_B/sep_slices.csv   # list typical slices
    python3 overlay_figs.py fig5 --root /path/to/CHAOS --out figs
    python3 overlay_figs.py fig5 ... --rows skin:30:z14,leak:1,pass2:33      # :zNN forces a slice

Writes fig2_ct_overlays.{pdf,png,tif} / fig5_mri_wall.{pdf,png,tif}, a _panels/ folder with
each panel at full size for checking, and a _check.txt that compares the recomputed pixel
counts with the run's CSV (fig2) or lists the per-slice figures used to pick slices (fig5).
Case keys only; both datasets are public.
"""
import argparse, csv, re, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('fig', choices=['fig2', 'fig5'])
ap.add_argument('--root', required=True)
ap.add_argument('--out', default='figs')
ap.add_argument('--dpi', type=int, default=600)
# fig2
ap.add_argument('--slices', default='patient_074:FILE200,patient_138:FILE126', help='fig2: comma list subject:slice')
ap.add_argument('--unet-run', default='runs/unet_f2m', help='fig2: run folder holding pred/<subject>/<slice>.png')
ap.add_argument('--sep-csv', default='', help='fig2: sep_slices.csv of the setting-B run, to check the recomputed counts')
ap.add_argument('--candidates', default='', help='fig2: sep_slices.csv - list typical slices of the even half and exit')
ap.add_argument('--fat-hi', type=float, default=79)
ap.add_argument('--cfat-hi', type=float, default=40)
ap.add_argument('--body', type=float, default=40)
ap.add_argument('--px', type=float, default=0.78)
ap.add_argument('--spacing', type=float, default=5.0)
ap.add_argument('--abd-up', type=int, default=0)
# fig5
ap.add_argument('--rows', default='skin:30,leak:1,pass2:33', help='fig5: mechanism:case[:zNN], comma separated')
ap.add_argument('--arm', default='mri_arm')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
pan = out / f'{a.fig}_panels'; pan.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Inter', 'DejaVu Sans'],
                     'font.size': 7, 'pdf.fonttype': 42})
C_SAT, C_VAT = np.array([0xf6, 0xc0, 0x8a]) / 255, np.array([0x9e, 0xc5, 0xf4]) / 255
C_V2S, C_S2V, C_NONE = np.array([0xe3, 0x49, 0x48]) / 255, np.array([0x62, 0x50, 0xd6]) / 255, np.array([0x1b, 0xaf, 0x7a]) / 255


def paint(gray, layers):
    o = np.dstack([gray] * 3).astype(float)
    for m, col, alpha in layers:
        if m is not None and m.any():
            o[m] = (1 - alpha) * o[m] + alpha * col
    return np.clip(o, 0, 1)


def split_layers(S, V, RS, RV, none=None):
    """S, V: the method's split; RS, RV: the reference's (all within the coloured fat)."""
    L = [(S & RS, C_SAT, 0.75), (V & RV, C_VAT, 0.75), (S & RV, C_V2S, 0.9), (V & RS, C_S2V, 0.9)]
    if none is not None:
        L.append((none, C_NONE, 0.9))
    return L


def crop_of(B, margin=6):
    rr = np.flatnonzero(B.any(1)); cc = np.flatnonzero(B.any(0))
    return (slice(max(rr[0] - margin, 0), rr[-1] + margin + 1), slice(max(cc[0] - margin, 0), cc[-1] + margin + 1))


def _rgb_tif(path):
    """matplotlib writes RGBA TIFFs; journals want RGB on white."""
    from PIL import Image
    im = Image.open(path)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
        bg.save(path, compression='tiff_lzw', dpi=im.info.get('dpi', (600, 600)))


def save(fig, stem):
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / f'{stem}.tif', dpi=a.dpi, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    _rgb_tif(out / f'{stem}.tif')


def legend(fig, items, y=0.005):
    fig.legend(handles=[Patch(facecolor=c, edgecolor='none', label=l) for c, l in items], loc='lower center',
               ncol=len(items), frameon=False, bbox_to_anchor=(0.5, y), fontsize=6.5, handlelength=1.2)


# ----------------------------------------------------------------------------------------
def fig2():
    if a.candidates:
        rows = list(csv.DictReader(open(a.candidates)))
        R = []
        for r in rows:
            n = int(re.findall(r'\d+', r['subject'])[-1])
            if n % 2:
                continue
            F = int(r['F_px'])
            R.append((r['subject'], r['slice'], F, int(r['ref_vat']) / F, 100 * int(r['meth_vat2sat']) / F, 100 * int(r['wall_sat2vat']) / F))
        mv = np.array([x[4] for x in R]); ws = np.array([x[5] for x in R])
        lo1, hi1, lo2, hi2 = np.percentile(mv, 75), np.percentile(mv, 92), np.percentile(ws, 70), np.percentile(ws, 92)
        print(f'even half: 2006 VAT->SAT (% of F) p75-p92 {lo1:.1f}-{hi1:.1f}; wall SAT->VAT p70-p92 {lo2:.1f}-{hi2:.1f}')
        print('subject      slice     F_px  VAT/F  2006 V->S  wall S->V')
        for x in sorted((x for x in R if lo1 <= x[4] <= hi1 and lo2 <= x[5] <= hi2), key=lambda x: x[2]):
            print(f'{x[0]:12} {x[1]:8} {x[2]:6d}  {x[3]:.2f}   {x[4]:6.2f}    {x[5]:6.2f}')
        return
    from fatvol.segment import est_fat_vol, SUBCUTANEOUS
    from wallsep import wall_zone
    WALL_KW = dict(close_mm=9.4, open_mm=0.78, skin_mm=3.0, anterior='up', open_post_mm=3.1)   # setting B
    root = Path(a.root).expanduser()
    run = Path(a.unet_run)
    want = [tuple(x.strip().split(':', 1)) for x in a.slices.split(',') if ':' in x]
    ref = {}
    if a.sep_csv:
        for r in csv.DictReader(open(a.sep_csv)):
            ref[(r['subject'], r['slice'])] = r

    def readlab(p):
        m = np.array(Image.open(p))
        return m.any(-1) if m.ndim == 3 else m > 0

    def find(d, stem):
        return next(iter(sorted(d.glob(stem + '.*'))), None)

    def num(p):
        d = re.findall(r'\d+', p.stem)
        return (int(d[-1]) if d else 0, p.stem)

    def shift(x, sy, sx):
        o = np.zeros_like(x); ny_, nx_ = x.shape[-2:]
        ys, yd = (slice(0, ny_ - sy), slice(sy, ny_)) if sy >= 0 else (slice(-sy, ny_), slice(0, ny_ + sy))
        xs, xd = (slice(0, nx_ - sx), slice(sx, nx_)) if sx >= 0 else (slice(-sx, nx_), slice(0, nx_ + sx))
        o[..., yd, xd] = x[..., ys, xs]
        return o

    panels, check = [], []
    for subj, sid in want:
        sd = root / 'Image' / 'Extracted' / subj
        G, S, V, ids = [], [], [], []
        for ip in sorted(sd.iterdir(), key=num):
            sp = find(root / 'Label' / 'Subcutaneous' / subj, ip.stem); vp = find(root / 'Label' / 'Visceral' / subj, ip.stem)
            if sp is None or vp is None:
                continue
            G.append(np.array(Image.open(ip).convert('L'))); S.append(readlab(sp)); V.append(readlab(vp)); ids.append(ip.stem)
        if sid not in ids:
            sys.exit(f'{subj}:{sid} not found')
        G, S, V = np.stack(G), np.stack(S), np.stack(V)
        B = np.stack([ndi.binary_fill_holes(ndi.binary_closing((g > a.body) | s | v, iterations=3)) for g, s, v in zip(G, S, V)])
        for i in range(len(B)):
            cc, n = ndi.label(B[i])
            if n:
                B[i] = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
        F = B & (G <= a.fat_hi) & (S | V)
        C = F & (G <= a.cfat_hi)
        rr = np.flatnonzero(B.any(axis=(0, 2))); cc_ = np.flatnonzero(B.any(axis=(0, 1)))
        dy = int(round((G.shape[1] - 1) / 2 - (rr[0] + rr[-1]) / 2)); dx = int(round((G.shape[2] - 1) / 2 - (cc_[0] + cc_[-1]) / 2))
        if shift(B, dy, dx).sum() != B.sum():
            dy = dx = 0
        T = lambda x: np.ascontiguousarray(np.transpose(shift(x, dy, dx), (2, 1, 0)))
        U = lambda x: shift(np.transpose(x, (2, 1, 0)), -dy, -dx)
        r = est_fat_vol(T(G).astype(np.float32), a.px, a.px, a.spacing, fat_loc=SUBCUTANEOUS,
                        fatmask_in=T(F), cfatmask_in=T(C), bdmask_in=T(B), abd_up=a.abd_up)
        i = ids.index(sid)
        MS = (U(r.fmask) > 0)[i] & F[i]; MV = (U(r.vfmask) > 0)[i] & F[i]
        WZ, _ = wall_zone(F[i], B[i], a.px, nonfat=B[i] & (G[i] > a.fat_hi), **WALL_KW)
        pp = run / 'pred' / subj / f'{sid}.png'
        if not pp.exists():
            sys.exit(f'no U-Net prediction {pp} (was the run made with --no-save-pred?)')
        pred = np.array(Image.open(pp))
        Fi, RS, RV = F[i], F[i] & S[i], F[i] & V[i]
        US, UV = Fi & (pred == 1), Fi & (pred == 2)
        splits = {'meth': (MS, MV, Fi & ~MS & ~MV), 'wall': (Fi & ~WZ, Fi & WZ, None), 'unet': (US, UV, Fi & ~US & ~UV)}
        line = f'{subj} {sid}: F {int(Fi.sum())}'
        for t, (s_, v_, _) in splits.items():
            line += f'  {t} VAT {int(v_.sum())} (V->S {int((RV & s_).sum())}, S->V {int((RS & v_).sum())})'
            k = (subj, sid)
            if k in ref and f'{t}_vat' in ref[k]:
                line += f' [csv {ref[k][t + "_vat"]}]'
        check.append(line); print(line)
        panels.append((subj, sid, G[i] / 255.0, B[i], Fi, RS, RV, splits))

    cols = [('meth', '2006 method'), ('wall', 'Wall rule, setting B'), ('unet', 'U-Net (f2m)')]
    crops = [crop_of(p[3]) for p in panels]
    asp = [(c[1].stop - c[1].start) / (c[0].stop - c[0].start) for c in crops]
    pw = 174 / 25.4 * 0.99 / (4 + 0.03 * 3)
    hs = [pw / x for x in asp]
    H = sum(hs) * (1 + 0.06) + 0.75
    fig = plt.figure(figsize=(174 / 25.4, H))
    gs = fig.add_gridspec(len(panels), 4, height_ratios=hs, hspace=0.06, wspace=0.03, left=0.005, right=0.995,
                          bottom=0.45 / H, top=1 - 0.3 / H)
    for rI, (subj, sid, g, B, Fi, RS, RV, splits) in enumerate(panels):
        cr = crops[rI]
        imgs = [(np.dstack([g] * 3), f'{subj.replace("patient_", "Subject ")}')]
        for t, title in cols:
            s_, v_, none = splits[t]
            mis = 100 * ((RV & s_).sum() + (RS & v_).sum()) / max(Fi.sum(), 1)
            imgs.append((paint(g, split_layers(s_, v_, RS, RV, none)), f'{mis:.1f}% of fat misassigned'))
        for cI, (im, lab) in enumerate(imgs):
            axx = fig.add_subplot(gs[rI, cI]); axx.imshow(im[cr], interpolation='nearest'); axx.axis('off')
            axx.text(0.02, 0.02, lab, transform=axx.transAxes, fontsize=6, color='white', va='bottom',
                     bbox=dict(boxstyle='square,pad=0.15', fc='black', ec='none', alpha=0.55))
            if rI == 0:
                axx.set_title(['Image', *[c[1] for c in cols]][cI], fontsize=7.5, fontweight='bold', pad=3)
            Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(pan / f'{subj}_{sid}_{["image", *[c[0] for c in cols]][cI]}.png')
    legend(fig, [(C_SAT, 'SAT, agreed'), (C_VAT, 'VAT, agreed'), (C_V2S, 'labelled VAT called SAT'),
                 (C_S2V, 'labelled SAT called VAT'), (C_NONE, 'neither compartment')])
    save(fig, 'fig2_ct_overlays')
    (out / 'fig2_check.txt').write_text('\n'.join(check) + '\n')
    print(f'-> {out}/fig2_ct_overlays.pdf/.png/.tif, {pan}/, fig2_check.txt')


# ----------------------------------------------------------------------------------------
def fig5():
    import nibabel as nib
    from fatvol.segment import _ridler_calvard
    from mri_body import body_trunk, lung_run, abdomen_slices, edge_cut
    from wallsep import wall_zone, skin_band, otsu3
    root = Path(a.root).expanduser(); arm = root / a.arm
    BASE = dict(open_mm=0.78, skin_mm=3.0, anterior='up', open_post_mm=3.1)
    CT = dict(BASE, close_mm=9.4, close_steps_mm=(15.0,), skin_max_mm=None, leak=True)
    MRI = dict(BASE, close_mm=25.0, close_steps_mm=(), skin_max_mm=10.0, leak=False)
    SAT_L, VAT_L = 65, 67
    s2 = lambda x: x.T[::-1, ::-1]

    def to_grid(seg_img, ref_img):
        if seg_img.shape == ref_img.shape and np.allclose(seg_img.affine, ref_img.affine, atol=1e-3):
            return np.asarray(seg_img.dataobj).astype(np.int16)
        from nibabel.processing import resample_from_to
        return np.asarray(resample_from_to(seg_img, (ref_img.shape, ref_img.affine), order=0).dataobj).astype(np.int16)

    canon = lambda arr, img: np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(arr, img.affine)).dataobj)
    rows, check = [], ['case mech z  ct_S->V_px ct_leak_px p1_V->S_px final_V->S_px final_S->V_px pass']
    for spec in a.rows.split(','):
        parts = spec.strip().split(':')
        mech, cid = parts[0], int(parts[1])
        zforce = int(parts[2][1:]) if len(parts) > 2 else None
        key = f'chaos_{cid:04d}'
        img = nib.load(str(arm / 'inputs' / f'{key}.nii.gz'))
        cimg = nib.as_closest_canonical(img)
        v = np.asanyarray(cimg.dataobj).astype(np.float32)
        v = v[..., 0] if v.ndim == 4 else v
        zx, zy, zz = (float(x) for x in cimg.header.get_zooms()[:3])
        seg = canon(to_grid(nib.load(str(arm / 'segs' / f'{key}_seg.nii.gz')), img), img).astype(np.int16)
        Tm, Am, notch, thr = body_trunk(v, min(zx, zy))
        lung = lung_run(v, Tm, thr, min(zx, zy))
        abd = abdomen_slices(Tm, lung)
        slab = [k for k in sorted(abd, reverse=True) if not edge_cut(Tm[:, :, k])]
        px = (zx + zy) / 2
        best, recs = None, []
        for k in slab:
            B, G, SEG = s2(Tm[:, :, k]), s2(v[:, :, k]), s2(seg[:, :, k])
            if not B.any():
                continue
            t = _ridler_calvard(G * B)
            F = B & (G >= t)
            BAND = skin_band(B & ~F, B, px, skin_max_mm=10.0)
            MSAT, MVAT = (SEG == SAT_L) & B & ~BAND, (SEG == VAT_L) & B
            zc, ic = wall_zone(F, B, px, nonfat=B & ~F, **CT)
            z1, i1 = wall_zone(F, B, px, nonfat=B & ~F, **MRI)
            high = B & (G < otsu3(G[B])[1])
            zf, iff = wall_zone(F, B, px, nonfat=B & ~F, nonfat_high=high, **MRI)
            m = dict(ct_s2v=int((F & MSAT & zc).sum()), ct_leak=int(ic.get('leak_moved_px', 0)),
                     p1_v2s=int((F & MVAT & ~z1).sum()), fin_v2s=int((F & MVAT & ~zf).sum()),
                     fin_s2v=int((F & MSAT & zf).sum()), passn=int(iff.get('pass', 1)))
            check.append(f"{key} {mech} {k} {m['ct_s2v']} {m['ct_leak']} {m['p1_v2s']} {m['fin_v2s']} {m['fin_s2v']} {m['passn']}")
            score = {'skin': m['ct_s2v'], 'leak': (m['ct_leak'], m['ct_s2v']), 'pass2': m['p1_v2s'] - m['fin_v2s']}[mech]
            rec = (score, k, G, B, F, MSAT, MVAT, zc, z1, zf)
            if (zforce is not None and k == zforce) or (zforce is None and (best is None or score > best[0])):
                best = rec
        if best is None:
            sys.exit(f'{key}: slice not found')
        print(f'{key} {mech}: slice z{best[1]}')
        rows.append((key, mech, zy / zx) + best[1:])

    titles = ['Image', 'Wall rule, CT setting', 'MRI setting, first pass', 'MRI setting, final', 'VIBESegmentator']
    mech_lab = {'skin': 'skin band as wall', 'leak': 'leak repair on a flank pad', 'pass2': 'open ring, second pass'}
    crops = [crop_of(r[5]) for r in rows]            # r[5] = the trunk mask B
    asp = [(c[1].stop - c[1].start) / ((c[0].stop - c[0].start) * r[2]) for c, r in zip(crops, rows)]
    pw = 174 / 25.4 * 0.99 / (5 + 0.03 * 4)
    hs = [pw / x for x in asp]
    H = sum(hs) * (1 + 0.06) + 0.75
    fig = plt.figure(figsize=(174 / 25.4, H))
    gs = fig.add_gridspec(len(rows), 5, height_ratios=hs, hspace=0.06, wspace=0.03, left=0.005, right=0.995,
                          bottom=0.45 / H, top=1 - 0.3 / H)
    for rI, (key, mech, aspect, k, G, B, F, MSAT, MVAT, zc, z1, zf) in enumerate(rows):
        cr = crops[rI]
        hi = np.percentile(G[B], 99.5)
        g = np.clip(G / hi, 0, 1)
        RS, RV = F & MSAT, F & MVAT
        ims = [(np.dstack([g] * 3), None, f'{key} z{k}\n{mech_lab.get(mech, mech)}')]
        for z in (zc, z1, zf):
            s2v = 100 * (RS & z).sum() / max(RS.sum(), 1); v2s = 100 * (RV & ~z).sum() / max(RV.sum(), 1)
            ims.append((paint(g, split_layers(F & ~z, F & z, RS, RV)), z, f'SAT→VAT {s2v:.0f}%\nVAT→SAT {v2s:.0f}%'))
        ims.append((paint(g, [(MSAT, C_SAT, 0.75), (MVAT, C_VAT, 0.75)]), None, 'model SAT\nand VAT'))
        for cI, (im, z, lab) in enumerate(ims):
            axx = fig.add_subplot(gs[rI, cI]); axx.imshow(im[cr], interpolation='nearest', aspect=aspect); axx.axis('off')
            if z is not None:
                axx.contour(z[cr].astype(float), [0.5], colors='white', linewidths=0.6)
            axx.text(0.02, 0.02, lab, transform=axx.transAxes, fontsize=5.5, color='white', va='bottom',
                     bbox=dict(boxstyle='square,pad=0.15', fc='black', ec='none', alpha=0.55))
            if rI == 0:
                axx.set_title(titles[cI], fontsize=7, fontweight='bold', pad=3)
            Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(pan / f'{key}_z{k}_{cI}.png')
    legend(fig, [(C_SAT, 'SAT, agreed'), (C_VAT, 'VAT, agreed'), (C_V2S, 'model VAT called SAT'),
                 (C_S2V, 'model SAT called VAT')])
    save(fig, 'fig5_mri_wall')
    (out / 'fig5_check.txt').write_text('\n'.join(check) + '\n')
    print(f'-> {out}/fig5_mri_wall.pdf/.png/.tif, {pan}/, fig5_check.txt')


fig2() if a.fig == 'fig2' else fig5()
