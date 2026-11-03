#!/usr/bin/env python3
"""aattct_unet.py - a 2D U-Net for the SAT/VAT split on AATTCT-IDS, scored like aattct_sep.py.

The geometric methods (2006 ellipse, wall) are scored in aattct_sep.py on how they split
one fat mask F into SAT and VAT.  This script trains a network on the annotators' labels
and scores it on exactly the same F, so the numbers sit side by side:

    F      = body & grey <= --fat-hi & (SAT label | VAT label)       (as in aattct_sep.py)
    target = 0 background, 1 SAT label, 2 VAT label (VAT wins where the labels overlap)
    split  = a pixel of F is VAT where the network's VAT score beats its SAT score

Cross-validation is by SUBJECT: 5 folds stratified on visceral fraction (folds.csv,
reused on later runs so separate --fold runs agree).  For each test fold the other four
train the network, 10% of their subjects are held back to pick the checkpoint (lowest
share of F misassigned), and the test fold is scored once with that checkpoint.

    aattct_unet.py --root /path/to/AATTCT-IDS --smoke --out runs/unet_smoke
    aattct_unet.py --root ... --fold 0 --out runs/unet            (one fold, ~1 h on a B580 at 512)
    aattct_unet.py --root ... --fold all --out runs/unet          (all five)
    aattct_unet.py --root ... --summary-only --out runs/unet      (re-print the combined table)

Outputs in --out: folds.csv; fold<k>/best.pt, train_log.csv, test_slices.csv, png/ overlays;
pred/<subject>/<slice>.png (native size, 0 bg / 1 SAT / 2 VAT); sep_slices_unet.csv and
sep_subjects_unet.csv over all finished folds (subject rows carry the same columns as
aattct_sep.py's sep_subjects.csv, prefixed unet_, so the two files merge on `subject`).
The printed summary has the same lines as aattct_sep.py, for all subjects and for the odd
and even halves (the wall method's settings were chosen on the odd half).

--mode f2 (Oct 2026) poses the network the same task the geometric methods get: it
receives F as a second input channel and is trained only on F's pixels, SAT against
VAT.  In the default --mode 3class the network must also decide what is body fat at
all; on slices with faint skin lines, streaks or external objects it called parts of F
"background", and the SAT/VAT choice for those pixels was then close to arbitrary.
Use --folds-from to put a new run on the folds of an earlier one.

--mode f2m (18-19 Oct 2026) is f2 with the image input restricted to the trunk: body
mask B, cut to the convex hull of the annotated region (SAT | VAT labels) grown by 4 px.
An arm resting against the flank (patient_179), the table and objects on the skin lie
outside the hull and vanish from the input, while everything inside the body outline
(organs, muscle, wall) stays.  The hull carries no SAT/VAT information beyond what F
already does.  (A first version filled the label union instead; in lean subjects such as
028 and 201 the thin SAT label is not a closed ring, the fill failed and the whole
interior of the body was blanked.)  At start-up the script reports how much of B each
slice lost, checks that no pixel of F is masked, and writes maskcheck/ pictures of the
slices that lost most (left: removed part magenta; right: the network's input).

Device: Intel XPU if present, else CUDA, else CPU (--device to force).  Images of
different sizes are padded to a square and resized to --size for the network; the
prediction is mapped back to the native grid before scoring.

Needs torch, numpy, scipy, pillow, matplotlib.  No MONAI.
"""
import argparse, csv, math, re, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--root', required=True)
ap.add_argument('--out', default='runs/unet')
ap.add_argument('--fold', default='all', help="'all' or a fold number")
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-4)
ap.add_argument('--size', type=int, default=512, help='network input size (multiple of 16)')
ap.add_argument('--base', type=int, default=32, help='channels at the first level')
ap.add_argument('--depth', type=int, default=5, help='levels (size must divide by 2^(depth-1))')
ap.add_argument('--val-frac', type=float, default=0.1, help='share of training subjects held back to pick the checkpoint')
ap.add_argument('--val-every', type=int, default=2)
ap.add_argument('--rot', type=float, default=10, help='augmentation: rotation (deg)')
ap.add_argument('--scale', type=float, default=0.10, help='augmentation: scale +/-')
ap.add_argument('--shift', type=float, default=0.05, help='augmentation: shift (fraction of size)')
ap.add_argument('--no-flip', action='store_true', help='augmentation: no left-right flips')
ap.add_argument('--tta', action='store_true', help='average with the left-right flipped prediction')
ap.add_argument('--amp', action='store_true', help='bfloat16 autocast')
ap.add_argument('--device', default='auto', choices=['auto', 'xpu', 'cuda', 'cpu'])
ap.add_argument('--fat-hi', type=float, default=79)
ap.add_argument('--body', type=float, default=40)
ap.add_argument('--png', type=int, default=10, help='overlay PNGs per fold')
ap.add_argument('--no-save-pred', action='store_true')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--smoke', action='store_true', help='10 subjects, fold 0, 2 epochs, size 128, tiny net: a pipeline check')
ap.add_argument('--summary-only', action='store_true')
ap.add_argument('--mode', default='3class', choices=['3class', 'f2', 'f2m'],
                help="3class: image in, background/SAT/VAT out; f2: image + F in, loss on F only (SAT vs VAT); "
                     "f2m: f2 with the image masked to the labelled trunk")
ap.add_argument('--folds-from', default='', help='folds.csv of an earlier run, so both runs use the same folds')
a = ap.parse_args()
if a.smoke:
    a.epochs, a.size, a.base, a.batch, a.fold, a.val_every = 2, 128, 8, 4, '0', 1
USE_F = a.mode in ('f2', 'f2m')
if a.size % (2 ** (a.depth - 1)):
    sys.exit(f'--size {a.size} must be divisible by {2 ** (a.depth - 1)}')

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if a.folds_from:
    src = Path(a.folds_from)
    if (out / 'folds.csv').exists() and (out / 'folds.csv').read_bytes() != src.read_bytes():
        sys.exit(f'{out}/folds.csv differs from {src}')
    (out / 'folds.csv').write_bytes(src.read_bytes())
root = Path(a.root).expanduser()


# --------------------------------------------------------------------------- scoring
def dice(x, y):
    s = x.sum() + y.sum()
    return 2.0 * (x & y).sum() / s if s else np.nan


def summarize(rows, srows, title):
    """The same lines as aattct_sep.py's summary, for the unet split."""
    if not rows:
        return
    t = 'unet'
    R = {k: np.array([x[k] for x in rows], float) for k in rows[0] if k not in ('subject', 'slice', 'fold')}
    ok = R['ref_vat'] > 0
    e = (R[f'{t}_vat'][ok] - R['ref_vat'][ok]) / R['ref_vat'][ok] * 100
    es = (R[f'{t}_sat'] - R['ref_sat']) / np.maximum(R['ref_sat'], 1) * 100
    print(f'\n{title}: {len(rows)} slices, {len(srows)} subjects')
    print(f'[{t}] per slice  VAT %err median {np.median(e):+.1f}  %RMSE {np.sqrt(np.mean(e**2)):.1f}   '
          f'SAT %err median {np.median(es):+.1f}  %RMSE {np.sqrt(np.mean(es**2)):.1f}')
    print(f'[{t}] Dice median  SAT {np.nanmedian(R[f"{t}_dice_sat"]):.3f}  VAT {np.nanmedian(R[f"{t}_dice_vat"]):.3f}')
    print(f'[{t}] of all F: ref VAT called SAT {R[f"{t}_vat2sat"].sum()/R["F_px"].sum():.2%}, '
          f'ref SAT called VAT {R[f"{t}_sat2vat"].sum()/R["F_px"].sum():.2%}, neither 0.00%')
    if len(srows) > 1:
        sv = np.array([s[f'{t}_vat'] for s in srows], float); rv = np.array([s['ref_vat'] for s in srows], float)
        ss = np.array([s[f'{t}_sat'] for s in srows], float); rs = np.array([s['ref_sat'] for s in srows], float)
        ev = (sv - rv) / rv * 100
        print(f'[{t}] per subject (summed slices)  VAT %RMSE {np.sqrt(np.mean(ev**2)):.1f}  '
              f'SAT %RMSE {np.sqrt(np.mean(((ss-rs)/rs*100)**2)):.1f}  r(VAT) {np.corrcoef(sv, rv)[0,1]:.3f}')
        print(f'[{t}] per subject  VAT %err median {np.median(ev):+.1f}  subjects off by >5%: {int((abs(ev) > 5).sum())}  '
              f'>10%: {int((abs(ev) > 10).sum())}  of {len(srows)}')


def subject_rows(rows):
    by = {}
    for x in rows:
        by.setdefault(x['subject'], []).append(x)
    srows = []
    for s in sorted(by):
        sub = by[s]
        tot = lambda k: sum(x[k] for x in sub)
        srows.append(dict(subject=s, fold=sub[0]['fold'], n_slices=len(sub), ref_sat=tot('ref_sat'), ref_vat=tot('ref_vat'),
                          unet_sat=tot('unet_sat'), unet_vat=tot('unet_vat'), unet_none=0,
                          unet_vat2sat=tot('unet_vat2sat'), unet_sat2vat=tot('unet_sat2vat')))
    return srows


def snum(s):
    d = re.findall(r'\d+', s)
    return int(d[-1]) if d else 0


def write_csv(path, data):
    keys = list(dict.fromkeys(k for d in data for k in d))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(data)


def read_csv(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ('subject', 'slice'):
                r[k] = float(v) if ('dice' in k or v in ('', 'nan')) else int(v)
    return rows


def combined_summary():
    rows = []
    done = []
    for k in range(a.folds):
        p = out / f'fold{k}' / 'test_slices.csv'
        if p.exists():
            rows += read_csv(p); done.append(k)
    if not rows:
        print('no finished folds yet'); return
    srows = subject_rows(rows)
    write_csv(out / 'sep_slices_unet.csv', rows)
    write_csv(out / 'sep_subjects_unet.csv', srows)
    print(f'\n=== combined over folds {done} ===')
    summarize(rows, srows, 'all subjects')
    for half, par in (('odd half', 1), ('even half', 0)):
        summarize([r for r in rows if snum(r['subject']) % 2 == par],
                  [s for s in srows if snum(s['subject']) % 2 == par], half)
    print(f'-> {out}/sep_slices_unet.csv, sep_subjects_unet.csv')


if a.summary_only:
    combined_summary(); sys.exit(0)

import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.manual_seed(a.seed)
if a.device == 'auto':
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        dev = torch.device('xpu')
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
else:
    dev = torch.device(a.device)


# --------------------------------------------------------------------------- data
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


def body_mask(g, s, v):
    """As aattct_sep.py: filled skin contour united with the labels, largest component."""
    b = ndi.binary_fill_holes(ndi.binary_closing((g > a.body) | s | v, iterations=3))
    cc, n = ndi.label(b)
    if n:
        b = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
    return b


def to_net(arr, side, size, resample):
    """Pad (centred) to side x side, then resize to size x size."""
    h, w = arr.shape
    py, px = (side - h) // 2, (side - w) // 2
    sq = np.zeros((side, side), arr.dtype); sq[py:py + h, px:px + w] = arr
    if side == size:
        return sq, (py, px)
    return np.array(Image.fromarray(sq).resize((size, size), resample)), (py, px)


subj_dirs = sorted(p for p in (root / 'Image' / 'Extracted').iterdir() if p.is_dir())
if a.smoke:
    subj_dirs = subj_dirs[:10]
t0 = time.time()
Xn, Tn, Fn_, CODE, META = [], [], [], [], []  # network image / target / F; native F code; (subject, slice, side, pad)
overlap = 0
masked = []                                   # (pixels of B removed by the f2m trunk mask, subject, slice)
mask_pics = {}


def trunk_mask(b, s, v):
    """B cut to the convex hull of the labelled region, grown by 4 px (f2m)."""
    from scipy.spatial import ConvexHull
    from PIL import ImageDraw
    pts = np.argwhere(s | v)
    if len(pts) < 3:
        return b
    hull = pts[ConvexHull(pts).vertices]
    im = Image.new('1', (b.shape[1], b.shape[0]), 0)
    ImageDraw.Draw(im).polygon([(int(c), int(r)) for r, c in hull], fill=1, outline=1)
    return b & ndi.binary_dilation(np.array(im, bool), iterations=4)
for sd in subj_dirs:
    for ip in sorted(sd.iterdir(), key=num):
        sp = find(root / 'Label' / 'Subcutaneous' / sd.name, ip.stem)
        vp = find(root / 'Label' / 'Visceral' / sd.name, ip.stem)
        if sp is None or vp is None:
            continue
        g = np.array(Image.open(ip).convert('L')); s = readlab(sp); v = readlab(vp)
        b = body_mask(g, s, v)
        F = b & (g <= a.fat_hi) & (s | v)
        code = (F & s).astype(np.uint8) + 2 * (F & v).astype(np.uint8)       # 1 SAT, 2 VAT, 3 both (as aattct_sep counts both)
        overlap += int((s & v).sum())
        t = np.where(v, 2, np.where(s, 1, 0)).astype(np.uint8)
        side = max(g.shape)
        if a.mode == 'f2m':
            trunk = trunk_mask(b, s, v)
            if (F & ~trunk).any():
                sys.exit(f'{sd.name}:{ip.stem}: the trunk mask would hide {int((F & ~trunk).sum())} pixels of F')
            nrem = int((b & ~trunk).sum())
            masked.append((nrem, sd.name, ip.stem))
            if nrem > 2000:
                mask_pics[(sd.name, ip.stem)] = (g, b & ~trunk, trunk)
                if len(mask_pics) > 40:                      # keep the 20 largest in memory
                    for kk in sorted(mask_pics, key=lambda q: int(mask_pics[q][1].sum()))[:20]:
                        del mask_pics[kk]
            g = np.where(trunk, g, 0).astype(g.dtype)
        gx, pad = to_net(g, side, a.size, Image.BILINEAR)
        tx, _ = to_net(t, side, a.size, Image.NEAREST)
        if USE_F:
            fx, _ = to_net(F.astype(np.uint8), side, a.size, Image.NEAREST); Fn_.append(fx)
        Xn.append(gx); Tn.append(tx); CODE.append(code); META.append((sd.name, ip.stem, side, pad))
if not META:
    sys.exit('no labelled slices found under ' + str(root))
Xn = np.stack(Xn); Tn = np.stack(Tn); FN = np.stack(Fn_) if Fn_ else None
SUBJ = np.array([m[0] for m in META])
subjects = sorted(set(SUBJ))
print(f'{len(META)} slices, {len(subjects)} subjects loaded in {time.time()-t0:.0f}s; '
      f'device {dev}; SAT/VAT label overlap {overlap} px', flush=True)
if masked:
    mv = np.array([m[0] for m in masked])
    top = sorted(masked, reverse=True)[:5]
    print(f'f2m trunk mask: body pixels removed per slice median {int(np.median(mv))}, '
          f'slices losing >2000: {int((mv > 2000).sum())}; largest: '
          + ', '.join(f'{s_}:{f_} {n_}' for n_, s_, f_ in top), flush=True)
    md = out / 'maskcheck'; md.mkdir(exist_ok=True)
    for (s_, f_), (g_, rem, tr) in sorted(mask_pics.items(), key=lambda q: -int(q[1][1].sum()))[:8]:
        left = np.dstack([g_] * 3).copy(); left[rem] = (255, 0, 255)
        right = np.dstack([np.where(tr, g_, 0)] * 3)
        Image.fromarray(np.hstack([left, right]).astype(np.uint8)).save(md / f'{s_}_{f_}.png')
    print(f'-> {md}/ (left: removed part magenta; right: the network input)', flush=True)

# folds: stratified on visceral fraction, kept in folds.csv
fpath = out / 'folds.csv'
fold_of = {}
if fpath.exists():
    with open(fpath) as fh:
        for r in csv.DictReader(fh):
            fold_of[r['subject']] = int(r['fold'])
missing = [s for s in subjects if s not in fold_of]
if missing:
    if fold_of:
        sys.exit(f'{fpath} lacks {len(missing)} subjects (e.g. {missing[0]}); use another --out or delete it')
    vf = {}
    for s in subjects:
        idx = np.flatnonzero(SUBJ == s)
        sat = sum(int(np.isin(CODE[i], (1, 3)).sum()) for i in idx)
        vat = sum(int(np.isin(CODE[i], (2, 3)).sum()) for i in idx)
        vf[s] = vat / max(sat + vat, 1)
    order = sorted(subjects, key=lambda s: vf[s])
    rng = np.random.default_rng(a.seed)
    for i in range(0, len(order), a.folds):
        blk = order[i:i + a.folds]
        labels = rng.permutation(a.folds)[:len(blk)]
        for s, f in zip(blk, labels):
            fold_of[s] = int(f)
    write_csv(fpath, [dict(subject=s, fold=fold_of[s], visceral_fraction=round(vf[s], 4)) for s in subjects])
FOLD = np.array([fold_of[s] for s in SUBJ])


# --------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                               nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, base=32, depth=5, n_out=3, n_in=1):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        ci = n_in
        for c in ch:
            self.down.append(Block(ci, c)); ci = c
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(ci, c, 2, stride=2)); self.dec.append(Block(2 * c, c)); ci = c
        self.head = nn.Conv2d(ci, n_out, 1)

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.down):
            x = b(x)
            if i < len(self.down) - 1:
                skips.append(x); x = Fn.max_pool2d(x, 2)
        for u, d in zip(self.up, self.dec):
            x = d(torch.cat([u(x), skips.pop()], 1))
        return self.head(x)


def batch_tensors(idx):
    """Image (B,1,H,W) in 0..1, F (B,1,H,W) 0/1 or None, target (B,H,W)."""
    x = torch.from_numpy(Xn[idx]).to(dev).float().div_(255).unsqueeze(1)
    f = torch.from_numpy(FN[idx]).to(dev).float().unsqueeze(1) if FN is not None else None
    t = torch.from_numpy(Tn[idx]).to(dev).long()
    return x, f, t


def net_in(x, f):
    return x if f is None else torch.cat([x, f], 1)


def augment(x, f, t):
    n = x.shape[0]
    u = lambda: torch.rand(n, device=dev) * 2 - 1
    ang = u() * math.radians(a.rot)
    sc = 1 + u() * a.scale
    sx = torch.ones(n, device=dev) if a.no_flip else 1 - 2 * (torch.rand(n, device=dev) < .5).float()
    c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([c * sx, -s, u() * a.shift * 2], 1),
                         torch.stack([s * sx, c, u() * a.shift * 2], 1)], 1)
    grid = Fn.affine_grid(theta, list(x.shape), align_corners=False)
    x = Fn.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    t = Fn.grid_sample(t.unsqueeze(1).float(), grid, mode='nearest', padding_mode='zeros',
                       align_corners=False).squeeze(1).long()
    if f is not None:
        f = Fn.grid_sample(f, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    gam = torch.exp(u() * math.log(1.3)).view(-1, 1, 1, 1)
    x = x.clamp(0, 1) ** gam
    x = (x - .5) * (1 + .1 * u()).view(-1, 1, 1, 1) + .5 + .05 * u().view(-1, 1, 1, 1)
    x = x + .01 * torch.randn_like(x)
    return x, f, t


def loss_f2(logits, t, f):
    """SAT against VAT on the pixels of F only (logit channels 1 and 2)."""
    lo = logits[:, 1:]
    tt = (t - 1).clamp(min=0)                                # 0 SAT, 1 VAT
    m = (f[:, 0] > .5) & (t > 0)
    if not m.any():
        return lo.sum() * 0
    ce = Fn.cross_entropy(lo, tt, reduction='none')[m].mean()
    p = lo.softmax(1); mm = m.unsqueeze(1).float()
    oh = Fn.one_hot(tt, 2).permute(0, 3, 1, 2).float()
    inter = (p * oh * mm).sum((0, 2, 3)); den = (p * mm).sum((0, 2, 3)) + (oh * mm).sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (den + 1)).mean()


def loss_fn(logits, t):
    ce = Fn.cross_entropy(logits, t)
    p = logits.softmax(1)
    oh = Fn.one_hot(t, 3).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


@torch.no_grad()
def predict(model, idx_list, bs):
    """Yield (i, label map at native size: 0 bg / 1 SAT / 2 VAT, VAT-beats-SAT map at native size)."""
    model.eval()
    for j in range(0, len(idx_list), bs):
        idx = idx_list[j:j + bs]
        x, f, _ = batch_tensors(idx)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp):
            lo = model(net_in(x, f)).float()
            if a.tta:
                lo = (lo + model(net_in(x.flip(-1), None if f is None else f.flip(-1))).float().flip(-1)) / 2
        for k, i in enumerate(idx):
            _, _, side, (py, px) = META[i]
            h, w = CODE[i].shape
            l = lo[k:k + 1]
            if side != a.size:
                l = Fn.interpolate(l, size=(side, side), mode='bilinear', align_corners=False)
            l = l[0, :, py:py + h, px:px + w]
            vat = (l[2] > l[1]).cpu().numpy()
            lab = (np.where(CODE[i] > 0, 1 + vat, 0).astype(np.uint8) if USE_F
                   else l.argmax(0).cpu().numpy().astype(np.uint8))
            yield i, lab, vat


def score(i, vat, fold):
    code = CODE[i]
    F = code > 0
    RS, RV = np.isin(code, (1, 3)), np.isin(code, (2, 3))
    ms, mv = F & ~vat, F & vat
    return dict(subject=META[i][0], slice=META[i][1], fold=fold, F_px=int(F.sum()), ref_sat=int(RS.sum()), ref_vat=int(RV.sum()),
                unet_sat=int(ms.sum()), unet_vat=int(mv.sum()), unet_none=0,
                unet_vat2sat=int((RV & ms).sum()), unet_sat2vat=int((RS & mv).sum()),
                unet_dice_sat=round(dice(ms, RS), 4), unet_dice_vat=round(dice(mv, RV), 4))


def overlay(i, vat, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    g = np.array(Image.open(next((root / 'Image' / 'Extracted' / META[i][0]).glob(META[i][1] + '.*'))).convert('L'))
    code = CODE[i]; F = code > 0; RS, RV = np.isin(code, (1, 3)), np.isin(code, (2, 3))
    gray = np.dstack([g / 255.0] * 3); ov = gray.copy()
    OS, OV = F & ~vat, F & vat
    for m, c in ((RS & OS, (1, .55, 0)), (RV & OV, (0, .8, 1)), (RV & OS, (1, 0, 0)), (RS & OV, (0, 0, 1))):
        ov[m] = .35 * gray[m] + .65 * np.array(c)
    rv = int(RV.sum()) or 1; uv = int(OV.sum())
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    ax[0].imshow(gray); ax[1].imshow(ov)
    for x in ax: x.axis('off')
    ax[0].set_title(f'{META[i][0]} {META[i][1]}', fontsize=9)
    ax[1].set_title(f'VAT ref {rv} unet {uv} ({(uv-rv)/rv*100:+.1f}%)  red: ref VAT->SAT  blue: ref SAT->VAT', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# --------------------------------------------------------------------------- one fold
def run_fold(k):
    fd = out / f'fold{k}'; fd.mkdir(exist_ok=True)
    test_idx = np.flatnonzero(FOLD == k)
    pool = sorted(set(SUBJ[FOLD != k]))
    rng = np.random.default_rng(a.seed + 100 + k)
    n_val = int(round(a.val_frac * len(pool))) if len(pool) >= 5 else 0
    val_s = set(rng.choice(pool, n_val, replace=False)) if n_val else set()
    tr_idx = np.flatnonzero((FOLD != k) & ~np.isin(SUBJ, list(val_s)))
    va_idx = np.flatnonzero(np.isin(SUBJ, list(val_s)))
    print(f'\nfold {k}: train {len(tr_idx)} slices / {len(pool) - n_val} subjects, val {len(va_idx)} / {n_val}, '
          f'test {len(test_idx)} / {len(set(SUBJ[test_idx]))}', flush=True)

    torch.manual_seed(a.seed + k)
    model = UNet(a.base, a.depth, n_in=2 if USE_F else 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = math.ceil(len(tr_idx) / a.batch); total = steps * a.epochs; warm = min(200, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(st, total) / total)))
    best, log = None, []
    for ep in range(a.epochs):
        model.train(); te = time.time(); losses = []
        perm = rng.permutation(tr_idx)
        for b in range(steps):
            x, f, t = batch_tensors(perm[b * a.batch:(b + 1) * a.batch])
            x, f, t = augment(x, f, t)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp):
                lo = model(net_in(x, f))
            loss = loss_f2(lo.float(), t, f) if USE_F else loss_fn(lo.float(), t)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        rec = dict(epoch=ep + 1, loss=round(float(np.mean(losses)), 5), seconds=round(time.time() - te, 1))
        if len(va_idx) and ((ep + 1) % a.val_every == 0 or ep + 1 == a.epochs):
            vr = [score(i, vat, k) for i, _, vat in predict(model, va_idx, a.batch)]
            mis = 100 * sum(r['unet_vat2sat'] + r['unet_sat2vat'] for r in vr) / max(sum(r['F_px'] for r in vr), 1)
            rec['val_misassigned_pct'] = round(mis, 4)
            if best is None or mis < best:
                best = mis; torch.save(model.state_dict(), fd / 'best.pt'); rec['saved'] = 1
        log.append(rec)
        print(f"fold {k} epoch {ep+1}/{a.epochs}  loss {rec['loss']:.4f}"
              + (f"  val misassigned {rec['val_misassigned_pct']:.2f}%" if 'val_misassigned_pct' in rec else '')
              + (' *' if rec.get('saved') else '') + f"  {rec['seconds']:.0f}s", flush=True)
        write_csv(fd / 'train_log.csv', log)
    if best is None:
        torch.save(model.state_dict(), fd / 'best.pt')
    model.load_state_dict(torch.load(fd / 'best.pt', map_location=dev))
    rows, n_png = [], 0
    (fd / 'png').mkdir(exist_ok=True)
    for i, lab, vat in predict(model, test_idx, a.batch):
        rows.append(score(i, vat, k))
        if not a.no_save_pred:
            pd_ = out / 'pred' / META[i][0]; pd_.mkdir(parents=True, exist_ok=True)
            Image.fromarray(lab).save(pd_ / f'{META[i][1]}.png')
        if n_png < a.png:
            overlay(i, vat, fd / 'png' / f'{META[i][0]}_{META[i][1]}.png'); n_png += 1
    write_csv(fd / 'test_slices.csv', rows)
    summarize(rows, subject_rows(rows), f'fold {k} test')


folds = range(a.folds) if a.fold == 'all' else [int(a.fold)]
(out / 'params.txt').write_text(' '.join(sys.argv) + '\n' + '\n'.join(f'{k} = {v}' for k, v in sorted(vars(a).items())) + '\n')
for k in folds:
    run_fold(k)
combined_summary()
