#!/usr/bin/env python3
"""totalseg_aattct.py - TotalSegmentator's tissue_types task on AATCT-IDS, scored like
the other CT methods (same fat mask F, same annotators, same odd/even halves).

AATCT-IDS ships 8-bit JPEGs, not Hounsfield units.  aattct_probe_hu.py (14 Nov 2026,
12 subjects) measured the window actually used: fat median at grey 6 with 36% of fat
pixels clipped to 0, soft tissue median 152, bone clipped at 255, so roughly
lo = -106 HU, hi = +147 HU (not the paper's [-1000, +1000]).  This script therefore
gives TotalSegmentator *reconstructed* values:

    HU = lo + grey * (hi - lo) / 255      inside the body
    HU = -1000                            outside the body (air, table)

Fat below lo, bowel gas and air inside the body all come out at lo, and bone stops at
hi; the paper must say TotalSegmentator saw these reconstructed values, not the
original CT.  The other methods saw the same JPEGs.

Steps (each skips subjects already done, so it can be stopped and resumed):

  prep   build one NIfTI per subject from its full Initial series (consecutive
         1.25 mm slices), pixel size --px (unknown after cropping; 0.78 mm as in
         aattct_sep.py), slice order --z (file number increasing = 'down' = towards
         the feet, or 'up')
  run    TotalSegmentator -ta tissue_types --ml on each NIfTI (needs the academic
         licence: totalseg_set_license -l aca_...)
  score  for every annotated (Extracted) slice: F = body & grey <= --fat-hi &
         (SAT label | VAT label), exactly as aattct_sep.py; TotalSegmentator's
         subcutaneous_fat = SAT, torso_fat = VAT, anything else = neither
  png    overlays for a few slices, to check orientation and plausibility
  check  before 'run': share of annotated fat outside the body mask, and the
         reconstructed HU of one slice per subject drawn from the NIfTI in patient
         orientation (liver must be on the left of the picture)

    # pilot: 3 subjects, both slice orders, compare the score summaries
    python totalseg_aattct.py prep  --root /data/LCClinic/fat/AATTCT-IDS --out ts_down --z down --subjects patient_001,patient_002,patient_003
    python totalseg_aattct.py run   --out ts_down
    python totalseg_aattct.py score --root /data/LCClinic/fat/AATTCT-IDS --out ts_down
    python totalseg_aattct.py png   --root /data/LCClinic/fat/AATTCT-IDS --out ts_down --n 4
    (repeat with --out ts_up --z up; keep the order that agrees better, then run all 300)

Outputs in --out: nifti/, seg/, ts_slices.csv (one row per annotated slice),
ts_subjects.csv (one row per subject), ts_summary.txt (Table 3 measures by half),
params_<step>.txt, png/.
"""
import argparse, csv, re, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument('step', choices=['prep', 'run', 'score', 'png', 'check'])
ap.add_argument('--root', default='/data/LCClinic/fat/AATTCT-IDS')
ap.add_argument('--out', default='ts_aattct')
ap.add_argument('--subjects', default='', help='comma list of patient_NNN (default: all)')
ap.add_argument('--lo', type=float, default=-106.0, help='HU at grey 0 (aattct_probe_hu.py fit)')
ap.add_argument('--hi', type=float, default=147.0, help='HU at grey 255')
ap.add_argument('--px', type=float, default=0.78, help='assumed pixel size, mm')
ap.add_argument('--dz', type=float, default=1.25, help='slice spacing of the Initial series, mm')
ap.add_argument('--z', default='down', choices=['down', 'up'],
                help="file number increasing = 'down' (towards the feet) or 'up'")
ap.add_argument('--body', type=float, default=40, help='body = filled (grey > this), as aattct_sep.py')
ap.add_argument('--with-labels', action='store_true',
                help="prep: complete the body outline with the annotators' SAT|VAT of the nearest annotated "
                     "slice (<= 2 files away), as aattct_sep.py does for the other CT methods")
ap.add_argument('--fat-hi', type=float, default=79, help='F = grey <= this, as aattct_sep.py')
ap.add_argument('--device', default='cpu')
ap.add_argument('--task', default='tissue_types')
ap.add_argument('--n', type=int, default=4, help='png: number of subjects')
a = ap.parse_args()

root, out = Path(a.root).expanduser(), Path(a.out)
(out / 'nifti').mkdir(parents=True, exist_ok=True)
(out / 'seg').mkdir(exist_ok=True)
(out / f'params_{a.step}.txt').write_text(' '.join(sys.argv) + '\n' +
                                           '\n'.join(f'{k} = {v}' for k, v in sorted(vars(a).items())) + '\n')

DEFAULT_LABELS = {1: 'subcutaneous_fat', 2: 'torso_fat', 3: 'skeletal_muscle'}


def num(p):
    d = re.findall(r'\d+', p.stem)
    return int(d[-1]) if d else -1


def grey(p):
    return np.asarray(Image.open(p).convert('L'), dtype=np.float32)


def lab(p):
    m = np.array(Image.open(p))
    return m.any(-1) if m.ndim == 3 else m > 0


def find(d, stem):
    for p in d.glob(stem + '.*'):
        return p
    return None


def body_mask(g, extra=None):
    """Same construction as aattct_sep.py's body (filled skin contour), labels not needed.
    For TotalSegmentator's input use a low --body (1 on AATCT-IDS): the air around the
    body is flat 0, while a faint skin line or grainy fat is not."""
    m0 = g > a.body
    if extra is not None:
        m0 = m0 | extra
    m = ndi.binary_closing(m0, iterations=3)
    m = ndi.binary_fill_holes(m)
    lb, n = ndi.label(m)
    if n > 1:                                   # keep the largest component (drops the table)
        sizes = ndi.sum(m, lb, range(1, n + 1))
        m = lb == (1 + int(np.argmax(sizes)))
    return m


def subjects():
    s = sorted(p.name for p in (root / 'Image' / 'Extracted').iterdir() if p.is_dir())
    if a.subjects:
        want = [x.strip() for x in a.subjects.split(',') if x.strip()]
        s = [x for x in s if x in want]
    return s


def labels_map():
    try:
        from totalsegmentator.map_to_binary import class_map
        return {int(k): v for k, v in class_map[a.task].items()}
    except Exception:
        return DEFAULT_LABELS


def affine():
    """Voxel (i = column, j = row, k = file order) -> RAS mm.  Images are displayed
    radiologically: column 0 = patient's right, row 0 = anterior (patients face up).
    RAS: +x = patient's left, so x grows with the column (+px); +y = anterior, so y
    falls with the row (-px); +z = superior, so 'down' (towards the feet) is -dz.
    (Before 14 Nov the x term was -px, which mirrored every volume left-right.)"""
    A = np.diag([a.px, -a.px, -a.dz if a.z == 'down' else a.dz, 1.0])
    return A


def prep():
    import nibabel as nib
    for s in subjects():
        dst = out / 'nifti' / f'{s}.nii.gz'
        if dst.exists():
            continue
        files = sorted((root / 'Image' / 'Initial' / s).glob('*.jp*g'), key=num)
        if not files:
            print(f'{s}: no Initial slices, skipped'); continue
        ann = {}                                    # file number -> annotated SAT|VAT
        if a.with_labels:
            for ip in (root / 'Image' / 'Extracted' / s).glob('*.jp*g'):
                sp = find(root / 'Label' / 'Subcutaneous' / s, ip.stem)
                vp = find(root / 'Label' / 'Visceral' / s, ip.stem)
                if sp is not None and vp is not None:
                    ann[num(ip)] = lab(sp) | lab(vp)
        vol = np.empty((512, 512, len(files)), np.int16)
        for k, f in enumerate(files):
            g = grey(f)
            extra = None
            if ann:
                near = min(ann, key=lambda n: abs(n - num(f)))
                if abs(near - num(f)) <= 2:
                    extra = ann[near]
            hu = a.lo + g * (a.hi - a.lo) / 255.0
            hu[~body_mask(g, extra)] = -1000
            vol[:, :, k] = np.round(hu).T          # (row, col) -> (i = col, j = row)
        img = nib.Nifti1Image(vol, affine())
        img.header.set_xyzt_units('mm')
        nib.save(img, str(dst))
        (out / 'nifti' / f'{s}.files.txt').write_text('\n'.join(f.stem for f in files) + '\n')
        print(f'{s}: {len(files)} slices -> {dst}')


def run():
    for src in sorted((out / 'nifti').glob('*.nii.gz')):
        s = src.name[:-7]
        if a.subjects and s not in a.subjects.split(','):
            continue
        dst = out / 'seg' / f'{s}.nii.gz'
        if dst.exists():
            continue
        t0 = time.time()
        cmd = ['TotalSegmentator', '-i', str(src), '-o', str(dst), '--ml', '-ta', a.task, '--device', a.device]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            print(f'{s}: FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}'); continue
        print(f'{s}: done in {time.time() - t0:.0f} s', flush=True)


def load_seg(s):
    import nibabel as nib
    p = out / 'seg' / f'{s}.nii.gz'
    if not p.exists():
        return None, None
    seg = np.asarray(nib.load(str(p)).dataobj).astype(np.int16)
    stems = (out / 'nifti' / f'{s}.files.txt').read_text().split()
    return seg, {st: k for k, st in enumerate(stems)}


def split_slice(seg, k, names):
    lab2d = seg[:, :, k].T                       # back to (row, col)
    inv = {v: kk for kk, v in names.items()}
    return lab2d == inv['subcutaneous_fat'], lab2d == inv['torso_fat']


def prep_z():
    """Slice order recorded by the prep step (score/png do not take --z themselves)."""
    p = out / 'params_prep.txt'
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('z = '):
                return line[4:].strip()
    return '?'


def score():
    names = labels_map()
    rows, subj = [], []
    for s in subjects():
        seg, idx = load_seg(s)
        if seg is None:
            continue
        tot = dict(S=0, V=0, PS=0, PV=0, F=0, s2v=0, v2s=0, nei=0)
        for ip in sorted((root / 'Image' / 'Extracted' / s).glob('*.jp*g'), key=num):
            sp = find(root / 'Label' / 'Subcutaneous' / s, ip.stem)
            vp = find(root / 'Label' / 'Visceral' / s, ip.stem)
            if sp is None or vp is None or ip.stem not in idx:
                continue
            g = grey(ip)
            S, V = lab(sp), lab(vp)
            body = body_mask(g) | S | V
            F = body & (g <= a.fat_hi) & (S | V)
            S, V = F & S, F & V
            PS, PV = split_slice(seg, idx[ip.stem], names)
            PS, PV = F & PS, F & PV & ~PS
            r = dict(subject=s, slice=ip.stem, F=int(F.sum()), S=int(S.sum()), V=int(V.sum()),
                     pred_S=int(PS.sum()), pred_V=int(PV.sum()),
                     sat_called_vat=int((S & PV).sum()), vat_called_sat=int((V & PS).sum()),
                     neither=int((F & ~PS & ~PV).sum()))
            rows.append(r)
            for k, c in (('S', 'S'), ('V', 'V'), ('PS', 'pred_S'), ('PV', 'pred_V'), ('F', 'F'),
                         ('s2v', 'sat_called_vat'), ('v2s', 'vat_called_sat'), ('nei', 'neither')):
                tot[k] += r[c]
        if tot['F']:
            subj.append(dict(subject=s, half='odd' if num(Path(s)) % 2 else 'even', **tot,
                             vat_err_pct=100 * (tot['PV'] - tot['V']) / tot['V'] if tot['V'] else np.nan,
                             sat_err_pct=100 * (tot['PS'] - tot['S']) / tot['S'] if tot['S'] else np.nan))
    if not rows:
        sys.exit('nothing to score: run prep and run first')
    for name, data in (('ts_slices.csv', rows), ('ts_subjects.csv', subj)):
        with open(out / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    lines = [f'TotalSegmentator {a.task} on AATCT-IDS, reconstructed HU [{a.lo:.0f}, {a.hi:.0f}], '
             f'px {a.px} mm, z {prep_z()} (prep); labels {names}',
             'half    n   VAT%RMSE  SAT%RMSE  F_mis%  SATcalledVAT%F  VATcalledSAT%F  neither%F  VATerr>5%  zero-mis']
    for half in ('odd', 'even', 'all'):
        ss = [x for x in subj if half == 'all' or x['half'] == half]
        if not ss:
            continue
        ve = np.array([x['vat_err_pct'] for x in ss], float)
        se = np.array([x['sat_err_pct'] for x in ss], float)
        F = sum(x['F'] for x in ss)
        s2v, v2s, nei = (sum(x[k] for x in ss) for k in ('s2v', 'v2s', 'nei'))
        lines.append(f'{half:5s} {len(ss):4d} {np.sqrt(np.nanmean(ve ** 2)):9.2f} {np.sqrt(np.nanmean(se ** 2)):9.2f} '
                     f'{100 * (s2v + v2s + nei) / F:7.3f} {100 * s2v / F:15.3f} {100 * v2s / F:15.3f} '
                     f'{100 * nei / F:10.3f} {int(np.sum(np.abs(ve) > 5)):10d} '
                     f'{sum(1 for x in ss if x["s2v"] + x["v2s"] + x["nei"] == 0):9d}')
    (out / 'ts_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def png():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = labels_map()
    (out / 'png').mkdir(exist_ok=True)
    done = 0
    for s in subjects():
        seg, idx = load_seg(s)
        if seg is None:
            continue
        ims = sorted((root / 'Image' / 'Extracted' / s).glob('*.jp*g'), key=num)
        for ip in (ims[0], ims[len(ims) // 2], ims[-1]):
            g = grey(ip)
            S = lab(find(root / 'Label' / 'Subcutaneous' / s, ip.stem))
            V = lab(find(root / 'Label' / 'Visceral' / s, ip.stem))
            PS, PV = split_slice(seg, idx[ip.stem], names)
            base = np.dstack([g / 255.0] * 3)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
            for x, (ss, vv, title) in zip(ax, ((S, V, 'annotators'), (PS, PV, 'TotalSegmentator'))):
                ov = base.copy()
                ov[ss] = .4 * ov[ss] + .6 * np.array([0.96, 0.75, 0.54])
                ov[vv] = .4 * ov[vv] + .6 * np.array([0.62, 0.77, 0.96])
                x.imshow(ov); x.set_title(f'{s} {ip.stem}: {title}', fontsize=9); x.axis('off')
            fig.tight_layout(); fig.savefig(out / 'png' / f'{s}_{ip.stem}.png', dpi=90); plt.close(fig)
        done += 1
        if done >= a.n:
            break
    print(f'{done} subjects -> {out / "png"}')


def check():
    """Before running TotalSegmentator: (1) share of the annotators' SAT/VAT pixels that
    the body mask leaves outside (those become -1000 HU, i.e. air, in the NIfTI);
    (2) a picture of the reconstructed HU of one annotated slice per subject, drawn in
    patient orientation from the NIfTI, so a left-right or front-back flip shows."""
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out / 'check').mkdir(exist_ok=True)
    for s in subjects():
        ims = sorted((root / 'Image' / 'Extracted' / s).glob('*.jp*g'), key=num)
        lost = tot = 0
        for ip in ims:
            sp = find(root / 'Label' / 'Subcutaneous' / s, ip.stem)
            vp = find(root / 'Label' / 'Visceral' / s, ip.stem)
            if sp is None or vp is None:
                continue
            g = grey(ip)
            L = lab(sp) | lab(vp)
            lost += int((L & ~body_mask(g, L if a.with_labels else None)).sum()); tot += int(L.sum())
        msg = f'{s}: annotated fat outside the body mask {100 * lost / max(tot, 1):.2f}%'
        src = out / 'nifti' / f'{s}.nii.gz'
        if src.exists() and ims:
            img = nib.as_closest_canonical(nib.load(str(src)))       # RAS
            v = np.asarray(img.dataobj)
            k = v.shape[2] // 2
            sl = np.rot90(v[:, :, k])        # RAS axial -> radiological view: patient's right on the left, anterior up
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(sl, cmap='gray', vmin=-200, vmax=200)
            ax.set_title(f'{s} middle slice, HU from NIfTI (radiological view)', fontsize=8)
            ax.text(5, 20, 'R', color='yellow'); ax.text(sl.shape[1] - 20, 20, 'L', color='yellow')
            ax.axis('off'); fig.tight_layout()
            fig.savefig(out / 'check' / f'{s}_hu.png', dpi=90); plt.close(fig)
            msg += f'; picture -> {out / "check" / (s + "_hu.png")}'
        print(msg)


{'prep': prep, 'run': run, 'score': score, 'png': png, 'check': check}[a.step]()
