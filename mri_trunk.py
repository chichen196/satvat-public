#!/usr/bin/env python3
"""mri_trunk.py - separate the trunk from the arms on an axial body mask (Oct 2026).

Used by chaos_mri_screen.py and meant for the MRI arm's preparation (2006 method, wall
rule): both need the trunk alone, and on CHAOS the arms often lie against the flank and
are cut off by the field of view.

trunk_mask(m, px, img=None, lr_axis=0) -> (trunk, info)

    m       filled body mask of one axial slice (bool, 2-D)
    px      pixel size (mm)
    img     the slice itself (optional): a piece is cut off as an arm only if at least
            --arm-nonfat of it is non-fat (muscle, bone), so a fold of flank fat is kept
    lr_axis the array axis that runs left-right (0 for a RAS [x, y] slice)

Steps
 1. Largest connected piece of the mask (arms lying free are separate pieces and go).
 2. Notches.  An arm pressed against the flank leaves two concave pockets in the outline,
    in front of the arm and behind it.  Pockets = convex hull minus mask; a pocket's tip is
    its point deepest inside the hull (the image edge counts as hull boundary).  Pockets
    deeper than notch_mm (12; 6 for a pocket that reaches the image edge, beside an arm
    mostly outside the field of view) whose tip lies in the outer part of the body (|lateral offset| >
    0.35 x half-width; the midline grooves of the back and the umbilicus are not) are
    candidates.  On each side every pair of candidates is tried as a cut (a 3-pixel line
    tip to tip); a cut is kept when the piece it frees lies lateral to the cut, covers
    between 1500 mm^2 and 30% of the body, is round rather than a thin cap (area / cut
    length >= 30 mm), and (with img) is at least arm_nonfat non-fat.  A piece that reaches
    the image edge is the part of an arm left inside the field of view: 500 mm^2 and
    10 mm are enough there.
    Of the cuts kept on a side, the shortest wins (the arm-trunk contact).
 3. Opening (fallback for an arm with no clear notch pair): opening of radius open_mm
    by distance transforms with the image edge as background, largest core, grown back
    open_mm inside the mask and then rec_mm more.  Arms are narrower than 2 x open_mm.
    A removed piece stays removed only if it looks like an arm (>= 1500 mm^2, outer body,
    and with img >= arm_nonfat non-fat); folds of flank fat go back to the trunk.
 The freed piece is not all arm: the trunk's own flank bulges past the straight cut.  The
 trunk outline radius about its centre, fitted (quadratic in angle) on both sides of the
 contact, is carried across it, and the piece's pixels inside that curve go back to the
 trunk (_restore_flank).  The true arm-trunk line (two skin layers) is not traced.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


def _poly_fill(shape, pts_rc):
    im = Image.new('1', (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([(float(c), float(r)) for r, c in pts_rc], fill=1, outline=1)
    return np.array(im, bool)


def _line(shape, p0, p1, width=3):
    im = Image.new('1', (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).line([(float(p0[1]), float(p0[0])), (float(p1[1]), float(p1[0]))], fill=1, width=width)
    return np.array(im, bool)


def _largest(m):
    cc, n = ndi.label(m)
    if not n:
        return m.copy()
    return cc == np.argmax(np.bincount(cc.ravel())[1:]) + 1


def hull_mask(m):
    pts = np.argwhere(m)
    if len(pts) < 3:
        return m.copy()
    try:
        h = ConvexHull(pts)
    except Exception:
        return m.copy()
    return _poly_fill(m.shape, pts[h.vertices]) | m


def _otsu(vals):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return np.nan
    hist, edges = np.histogram(vals, 256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _restore_flank(T0, P, tips, margin=np.radians(25), bin_w=np.radians(2.5)):
    """Pixels of the cut-off piece P that lie inside the trunk's own outline, carried across
    the arm contact.  The outline radius r(phi) about the trunk centre is taken from the
    boundary on both sides of the contact (margin wide), fitted with a quadratic in phi, and
    used inside the contact sector; the straight cut tip to tip would chord off the flank."""
    c = np.argwhere(T0).mean(0)
    B = T0 & ~ndi.binary_erosion(T0)
    b = np.argwhere(B) - c
    ang = np.arctan2(b[:, 1], b[:, 0]); rad = np.hypot(b[:, 0], b[:, 1])
    q = np.argwhere(P) - c
    aP = np.arctan2(q[:, 1].mean(), q[:, 0].mean())
    d = sorted(_wrap(np.arctan2(t[1] - c[1], t[0] - c[0]) - aP) for t in tips)
    lo, hi = d[0], d[-1]
    if not (lo < 0 < hi) or hi - lo > np.radians(120):
        return np.zeros_like(P)
    phi = _wrap(ang - aP)
    xs, ys = [], []
    for e0, e1 in ((lo - margin, lo), (hi, hi + margin)):
        for a0 in np.arange(e0, e1, bin_w):
            sel = (phi >= a0) & (phi < a0 + bin_w)
            if sel.any():
                xs.append(a0 + bin_w / 2); ys.append(rad[sel].max())
    if len(xs) < 6:
        return np.zeros_like(P)
    co = np.polyfit(xs, ys, 2)
    cap = 1.05 * max(ys)
    qphi = np.arctan2(q[:, 1], q[:, 0]) - aP
    qphi = _wrap(qphi)
    qr = np.hypot(q[:, 0], q[:, 1])
    inside = (qphi >= lo) & (qphi <= hi) & (qr <= np.minimum(np.polyval(co, qphi), cap))
    out = np.zeros_like(P)
    pts = np.argwhere(P)[inside]
    out[pts[:, 0], pts[:, 1]] = True
    return out


def trunk_mask(m, px, img=None, lr_axis=0, notch_mm=12.0, open_mm=40.0, rec_mm=10.0,
               arm_nonfat=0.25, min_arm_mm2=1500.0, max_arm_frac=0.30,
               min_thick_mm=30.0, min_thick_edge_mm=10.0, notch_edge_mm=6.0, min_arm_edge_mm2=500.0):
    info = dict(notch_cuts=0, opened=0)
    L = _largest(m)
    if L.sum() < 50:
        return L, info
    lat = np.indices(L.shape)[lr_axis]
    cl = lat[L].mean()
    hw = (lat[L].max() - lat[L].min()) / 2
    fat_t = _otsu(img[L].astype(float)) if img is not None else np.nan

    # 2. notches
    H = hull_mask(L)
    D = H & ~L
    depth = ndi.distance_transform_edt(np.pad(H, 1))[1:-1, 1:-1] * px
    cc, n = ndi.label(D)
    tips = []
    for i in range(1, n + 1):
        p = cc == i
        d = np.where(p, depth, 0)
        k = np.unravel_index(np.argmax(d), d.shape)
        at_edge = p[0].any() or p[-1].any() or p[:, 0].any() or p[:, -1].any()
        if d[k] >= (notch_edge_mm if at_edge else notch_mm) and abs(lat[k] - cl) > 0.35 * hw:
            tips.append(k)
    area_L = L.sum() * px * px
    removed = np.zeros_like(L)
    for side in (-1, 1):
        cand = [t for t in tips if np.sign(lat[t] - cl) == side]
        work = L & ~removed
        best = None
        for a_ in range(len(cand)):
            for b_ in range(a_ + 1, len(cand)):
                p0, p1 = cand[a_], cand[b_]
                cut = _line(L.shape, p0, p1)
                pieces, k = ndi.label(work & ~cut)
                if k < 2:
                    continue
                main = np.argmax(np.bincount(pieces.ravel())[1:]) + 1
                cut_lat = (lat[p0] + lat[p1]) / 2
                for j in range(1, k + 1):
                    if j == main:
                        continue
                    piece = pieces == j
                    a_mm = piece.sum() * px * px
                    edge = piece[0].any() or piece[-1].any() or piece[:, 0].any() or piece[:, -1].any()
                    if not ((min_arm_edge_mm2 if edge else min_arm_mm2) <= a_mm <= max_arm_frac * area_L):
                        continue
                    if side * (lat[piece].mean() - cut_lat) <= 0:          # must lie outside the cut
                        continue
                    if side * (lat[piece].mean() - cl) < 0.5 * hw:         # and in the outer body
                        continue
                    if img is not None and np.isfinite(fat_t) and (img[piece] < fat_t).mean() < arm_nonfat:
                        continue                                           # all fat: a flank fold, not an arm
                    length = np.hypot(*(np.array(p0, float) - np.array(p1, float)))
                    # an arm is round: area / contact length >= min_thick_mm; a slice of flank cut
                    # off along an irregular outline is a thin cap.  An arm cut off by the field of
                    # view is thinner, so a piece reaching the image edge needs only min_thick_edge_mm
                    if a_mm / max(length * px, 1) < (min_thick_edge_mm if edge else min_thick_mm):
                        continue
                    if best is None or length < best[0]:
                        best = (length, piece, (p0, p1))
        if best is not None:
            back = _restore_flank(work & ~best[1], best[1], best[2])
            removed |= best[1] & ~back
            info['notch_cuts'] += 1
    T = _largest(L & ~removed)

    # 3. opening fallback
    r, rr = open_mm / px, rec_mm / px
    cc, n = ndi.label(ndi.distance_transform_edt(np.pad(T, 1))[1:-1, 1:-1] > r)
    if n:
        core = cc == np.argmax(np.bincount(cc.ravel())[1:]) + 1
        t = T & (ndi.distance_transform_edt(~core) <= r)
        t = T & (ndi.distance_transform_edt(~t) <= rr)
        t = _largest(t)
        # keep only removals that look like arms: big enough, in the outer body, with muscle in
        # them; folds of flank fat, the back's bulges and small bumps go back to the trunk
        cc, n = ndi.label(T & ~t)
        for j in range(1, n + 1):
            piece = cc == j
            arm_like = (piece.sum() * px * px >= min_arm_mm2
                        and abs(lat[piece].mean() - cl) >= 0.5 * hw
                        and (img is None or not np.isfinite(fat_t) or (img[piece] < fat_t).mean() >= arm_nonfat))
            if not arm_like:
                t |= piece
        if t.sum() < T.sum():
            info['opened'] = int(T.sum() - t.sum())
        T = t
    return T, info
