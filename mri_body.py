#!/usr/bin/env python3
"""mri_body.py - body, trunk and lung masks for axial MRI volumes (Oct 2026).

The same steps chaos_mri_screen.py uses, as functions for the analysis scripts.
Arrays are RAS-canonical [x, y, z]: x toward the patient's right, y anterior, z superior.

    body_trunk(v, px, ...)  -> trunk (bool [x,y,z]), arms (bool), notch cuts per slice, thr
    lung_run(v, trunk, thr, px, ...) -> bool per slice: lung shows (followed down from the top)
    abdomen_slices(...)     -> indices of trunk slices below the lungs
    edge_cut(trunk_slice)   -> True when the trunk runs along an image edge (> 2% of it)
"""
import numpy as np
from scipy import ndimage as ndi

from mri_trunk import trunk_mask


def body_threshold(v, body_frac=0.08):
    return body_frac * float(np.percentile(v, 99.5))


def body_trunk(v, px, body_frac=0.08, median=5, arm_open=40.0, arm_rec=10.0):
    thr = body_threshold(v, body_frac)
    T = np.zeros(v.shape, bool); A = np.zeros(v.shape, bool); cuts = np.zeros(v.shape[2], int)
    st = ndi.generate_binary_structure(2, 1)
    for k in range(v.shape[2]):
        sl = ndi.median_filter(v[:, :, k], size=median) if median > 1 else v[:, :, k]
        m = np.pad(sl > thr, 4)
        m = ndi.binary_closing(m, st, iterations=2)[4:-4, 4:-4]
        m = ndi.binary_fill_holes(m)
        if not m.any():
            continue
        t, info = trunk_mask(m, px, img=v[:, :, k], lr_axis=0, open_mm=arm_open, rec_mm=arm_rec)
        rest = m & ~t
        cc, n = ndi.label(rest)
        if n:
            big = np.flatnonzero(np.bincount(cc.ravel())[1:] * px * px >= 500) + 1
            rest = np.isin(cc, big)
        T[:, :, k] = t; A[:, :, k] = rest; cuts[k] = info['notch_cuts']
    return T, A, cuts, thr


def lung_run(v, T, thr, px, lung_min=300.0, lung_frac=0.4):
    nz = v.shape[2]; flag = np.zeros(nz, bool)
    for k in range(nz):
        t = T[:, :, k]
        if t.sum() < 100:
            continue
        dthr = max(thr, lung_frac * np.median(v[:, :, k][t]))
        dark = ndi.binary_opening(t & (v[:, :, k] < dthr))
        cc, n = ndi.label(dark)
        if not n:
            continue
        xs, ys = np.nonzero(t)
        xc, ymid = np.median(xs), (ys.min() + ys.max()) / 2
        area = np.bincount(cc.ravel())[1:] * px * px
        sides = set()
        for i in np.flatnonzero(area >= lung_min) + 1:
            cx, cy = np.argwhere(cc == i).mean(0)
            if cy < ymid or area[i - 1] >= 10 * lung_min:
                sides.add(cx > xc)
        flag[k] = len(sides) == 2
    lung = np.zeros(nz, bool)
    has = np.flatnonzero(T.any((0, 1)))
    if not len(has):
        return lung
    upper = has[has >= has.max() - 0.4 * (has.max() - has.min())]
    start = [k for k in upper if flag[k]]
    if not start:
        return lung
    k, gap = max(start), 0
    while k >= 0 and gap <= 1:
        if flag[k]:
            lung[k] = True; gap = 0
        else:
            gap += 1
        k -= 1
    lung[max(start):] |= flag[max(start):]
    return lung


def abdomen_slices(T, lung):
    has = np.flatnonzero(T.any((0, 1)))
    low = np.flatnonzero(lung).min() if lung.any() else (has.max() + 1 if len(has) else 0)
    return has[has < low]


def edge_cut(t, frac=0.02):
    return bool(t[0].mean() > frac or t[-1].mean() > frac or t[:, 0].mean() > frac or t[:, -1].mean() > frac)
