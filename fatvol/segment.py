"""
Port of ESTFATVOLMRIV2 (C.-H. Chen, 2003-2007).

Estimates subcutaneous, visceral and total fat volume from an axial MRI volume.

The structure follows the .pro step for step so the two can be compared. Section
comments quote the original. Debug-only branches (the commented-out `for k=3,3`
single-slice loops, the `;;IsDisplay = 1` overrides) are dropped; interactive
display is replaced by an optional `on_step` callback so nothing blocks.

Volumes are (nx, ny, nz) — see idlcompat for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from .ellipse import fit_and_rasterise
from .idlcompat import (bdisk, closing, conditional_dilate, coordinate_masks,
                        dilate, erode, gray_threshold, mask_from_radial_profile,
                        morph_distance, opening, polar_outer_edge, sobel_edge)

SUBCUTANEOUS, VISCERAL, TOTAL = 0, 1, 2


@dataclass
class FatResult:
    """Everything the IDL function returned through keywords, plus the volume."""
    volume_ml: float
    fat_loc: int
    fmask: np.ndarray
    maskedge: np.ndarray
    vfmask: Optional[np.ndarray] = None
    vmaskedge: Optional[np.ndarray] = None
    is_abd_up: Optional[int] = None
    bdmask: Optional[np.ndarray] = None
    fatmask: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    # Oct 2026: the separation as a ZONE map, so it can be applied to any fat basis
    # (the RC mask, a shifted threshold, a fractional fat map).  uint8, (nx, ny, nz):
    # 0 outside the body mask, 1 body outside the visceral zone, 2 visceral zone
    # (rvisceromask, i.e. rvmdist > 0 in _visceral_pass).  Only set when fat_loc != TOTAL.
    # The method's own SAT/VAT labels are NOT a pure zone split: pfatmask is grown by
    # conditional dilation from the 1.3x seed mask, so some fat inside zone 2 is called
    # subcutaneous and some fat in zone 1 is in neither compartment.  Compare before use.
    zones: Optional[np.ndarray] = None
    notes: list = field(default_factory=list)


def _body_mask(images, pxsize, pysize, radmask, angmask, imask, jmask,
               is_arm_in, verbose):
    """'find the body part and fat mask' — body mask with arm removal and
    polar-coordinate edge repair."""
    nx, ny, nz = images.shape
    bdmask = np.zeros((nx, ny, nz), np.uint8)
    struo = bdisk(2, 5)
    struc = bdisk(round(nx / 20.0), round(nx / 10.0))
    bdmaskedger = np.zeros((nz, 360))

    for i in range(nz):
        sl = images[:, :, i]
        slavg = sl.sum() / float(nx * ny)
        tmp = (sl > slavg * 1.2).astype(np.uint8)

        # 'finding the arm position from profiles if necessary'
        if is_arm_in >= 1:
            prof = tmp.sum(axis=1)
            rax = int(np.argmax(prof[:nx // 6]))
            lax = int(np.argmax(prof[nx * 5 // 6:])) + nx * 5 // 6
            rbx = int(np.argmin(prof[rax:nx // 3])) + rax
            lbx = int(np.argmin(prof[nx * 2 // 3:lax + 1])) + nx * 2 // 3
            tmp[:rbx + 1, :] = 0
            tmp[lbx:, :] = 0

        tmp = opening(tmp, struo)           # remove small dots outside
        tmp2 = tmp.copy()
        tmp = closing(tmp, struc)           # fill gaps inside
        bdmask[:, :, i] = (tmp | tmp2)

        # edge in r-theta, about the IMAGE centre
        edge = sobel_edge(bdmask[:, :, i])
        r, _, _ = polar_outer_edge(edge, radmask, angmask)
        bdmaskedger[i] = r

        # degree-8 fit over a 1.5x wrapped copy, to avoid the -180/180 seam
        thi = np.arange(540)
        wrapped = np.concatenate([r, r[:180]])
        coef = np.polyfit(thi, wrapped, 8)
        bdfit = np.polyval(coef, thi)

        # 'use smoothness to check MR artifacts'
        diff = np.roll(r, -1) - np.roll(r, 1)
        diff[:3] = 0
        diff[357:] = 0
        imax, imin = diff.max(), diff.min()
        pidx = np.flatnonzero(diff >= imax * 0.5)
        nidx = np.flatnonzero(diff <= imin * 0.5)
        if pidx.size and nidx.size and pidx.size == nidx.size:
            if np.min(nidx - pidx) > 0:
                for a, b in zip(pidx, nidx):
                    bdmaskedger[i, a:b + 1] = bdfit[a:b + 1] * 1.1

                # 'remove outsider'
                keep = mask_from_radial_profile(bdmaskedger[i], radmask, angmask)
                keep = closing(bdmask[:, :, i] & keep, struc)
                bdmask[:, :, i] = bdmask[:, :, i] & keep

        if verbose:
            print(f"  body mask slice {i + 1}/{nz}", flush=True)

    return bdmask, bdmaskedger


def _idl_round(x):
    """IDL ROUND: nearest integer, halves away from zero."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def _idl_gray_threshold(data, thresh_method, nobins, idl_integer_division):
    """ESTFATVOLMRIV2.PRO lines 293-336 with IDL's actual HISTOGRAM behaviour on integer
    pixel data: the bin width (max-1)/(nbins-1) is truncated to an integer and values
    beyond min + width*(nbins-1) fall outside the histogram, while the index is converted
    back with the FLOAT width of line 294.  Kept to measure that departure (Table 2 of the
    accompanying paper); not the default.  `data` is the masked slice
    (or volume); masked-out pixels are 0 and excluded by MIN=1."""
    a = np.asarray(data).ravel()
    sel = a[a >= 1]
    if sel.size == 0:
        return 0
    maxpx = float(sel.max())
    bins = (maxpx - 1.0) / (nobins - 1)                       # line 294, float
    width = max(int(np.floor(bins)), 1)                       # IDL HISTOGRAM, integer input
    idx = ((sel.astype(np.int64) - 1) // width)
    keep = idx < nobins
    h = np.bincount(idx[keep], minlength=nobins)[:nobins].astype(np.int64)
    if thresh_method == 0:                                    # lines 298-309: valley between peaks
        imaxh = int(np.argmax(h))
        if imaxh >= 50:
            imaxh2 = int(np.argmax(h[0:50]))
            thresh = int(np.argmin(h[imaxh2:imaxh + 1])) + imaxh2
        else:
            imaxh2 = int(np.argmax(h[50:100]))
            thresh = int(np.argmin(h[imaxh:imaxh2 + 50 + 1])) + imaxh
    else:                                                     # lines 312-331: first-crossing intermeans
        moving, thresh = 0, float(nobins)
        ar = np.arange(nobins, dtype=np.int64)
        while moving + 1 <= thresh and moving <= nobins - 1:
            s1 = int((ar[:moving + 1] * h[:moving + 1]).sum()); s2 = int(h[:moving + 1].sum())
            s3 = int((ar[moving + 1:] * h[moving + 1:]).sum()); s4 = int(h[moving + 1:].sum())
            if idl_integer_division:
                q1 = 0 if s2 == 0 else s1 // s2; q3 = 0 if s4 == 0 else s3 // s4
            else:
                q1 = 0.0 if s2 == 0 else s1 / s2; q3 = 0.0 if s4 == 0 else s3 / s4
            thresh = q1 / 2.0 + q3 / 2.0
            moving += 1
    return _idl_round(bins * thresh)                          # line 336


def _ridler_calvard(data):
    """Plain Ridler-Calvard (ISODATA) intermeans on the unit-bin histogram of the body-masked
    pixels: start at the overall mean, iterate T = (mean below T + mean at/above T) / 2 to the
    fixed point.  Returns a float; pixels >= T are fat.  This is what the 2006 paper describes;
    the .pro transcription with IDL's integer histogram departs from it (see _idl_gray_threshold).
    Masked-out pixels are 0 and excluded."""
    a = np.asarray(data).ravel()
    sel = a[a >= 1].astype(np.float64)
    if sel.size == 0:
        return 0.0
    t = float(sel.mean())
    for _ in range(500):
        lo = sel[sel < t]; hi = sel[sel >= t]
        if lo.size == 0 or hi.size == 0:
            break
        nt = (lo.mean() + hi.mean()) / 2.0
        if abs(nt - t) < 1e-6:
            t = nt; break
        t = nt
    return t


def _threshold_fn(rule):
    """'rc' = plain Ridler-Calvard (default; the rule the 2006 paper describes);
    'idl' = ESTFATVOLMRIV2.PRO lines 293-336 with IDL's integer-histogram bin width;
    'float' = the same loop on a float-width histogram."""
    if rule == 'rc':
        return lambda data, thresh_method, nobins, idl_div: _ridler_calvard(data)
    if rule == 'idl':
        return _idl_gray_threshold
    if rule == 'float':
        return gray_threshold
    raise ValueError(f"unknown threshold rule {rule!r} (rc, idl, float)")


def _fat_mask(images, bdmask, thresh_method, nobins, is_vol_thresh,
              idl_integer_division, verbose, thresh_factor=1.0, threshold_rule='rc',
              thresh_mask_slice=None):
    """'use histograms of body part to define fat'.

    thresh_factor scales the threshold before the fat mask is taken; 1.0 is the .pro as
    ported and the validated setting.  It is kept for diagnostics only.
    The conservative seed mask keeps its own 1.3 (1.5 for the volume branch) on top."""
    nx, ny, nz = images.shape
    if is_vol_thresh:
        gt = _threshold_fn(threshold_rule)
        t = gt(images * bdmask, thresh_method, nobins, idl_integer_division) * thresh_factor
        if verbose:
            print(f"The threshold of gray level for whole body fat is {t:g}")
        fatmask = ((images >= t) & bdmask.astype(bool)).astype(np.uint8)
        cfatmask = ((images >= t * 1.5) & bdmask.astype(bool)).astype(np.uint8)
        return fatmask, cfatmask, np.full(nz, t)

    fatmask = np.zeros((nx, ny, nz), np.uint8)
    cfatmask = np.zeros((nx, ny, nz), np.uint8)
    thregray = np.zeros(nz, dtype=float)
    # Oct 2026: thresh_mask_slice = k histograms every slice inside slice k's body outline,
    # the 2003/2008 threshold program's line 176 (images[*,*,j]*bdmask, 2-D x 3-D, which IDL
    # truncates to bdmask[*,*,0]).  Only the histogram is affected; the fat mask is still
    # taken inside each slice's own body mask.  None (default) = each slice's own outline.
    hmask = None if thresh_mask_slice is None else bdmask[:, :, int(thresh_mask_slice)]
    for j in range(nz):
        gt = _threshold_fn(threshold_rule)
        hm = bdmask[:, :, j] if hmask is None else hmask
        t = gt(images[:, :, j] * hm, thresh_method, nobins, idl_integer_division) * thresh_factor
        thregray[j] = t
        if verbose:
            print(f"slice {j + 1}: threshold gray level {t:g}")
        fatmask[:, :, j] = (images[:, :, j] >= t) & bdmask[:, :, j].astype(bool)
        # note: 1.3 per-slice, but 1.5 for the whole-volume branch — as in the .pro
        cfatmask[:, :, j] = (images[:, :, j] >= t * 1.3) & bdmask[:, :, j].astype(bool)
    return fatmask, cfatmask, thregray


def _subcutaneous_ellipse_pass(fatmask, bdmask, pxsize, pysize,
                               radmask, angmask, xmask, ymask, notes, fit_rule='strict'):
    """Rough subcutaneous edge -> per-slice ellipse fit -> refined body mask."""
    nx, ny, nz = fatmask.shape
    struo, struc = bdisk(2, 5), bdisk(round(nx / 20.0), round(nx / 10.0))

    bdmdist = np.stack([morph_distance(bdmask[:, :, i]) for i in range(nz)], axis=2)
    rpfatmaskedge = ((bdmdist <= 15) & fatmask.astype(bool)).astype(np.uint8)

    pfmaskedger = np.zeros((nz, 360))
    pfatedgepara = np.zeros((nz, 5))
    ig0 = np.array([0, 0, 0,
                    nx / 4.0 * pxsize / 0.859375,
                    ny / 6.0 * pxsize / 0.859375], float)
    ig = ig0.copy()

    for k in range(nz):
        r, idxs, present = polar_outer_edge(rpfatmaskedge[:, :, k], radmask, angmask)
        pfmaskedger[k] = r

        oig = ig.copy()
        fitted, ig = fit_and_rasterise(rpfatmaskedge[:, :, k], ig,
                                       radmask, angmask, xmask, ymask, pxsize, fit_rule)
        # the .pro's own sanity check on the fit
        if (np.max(np.abs(oig[0:2] - ig[0:2])) > nx / 8 * pxsize
                or np.max(np.abs(oig[3:5] - ig[3:5]) / np.maximum(oig[3:5], 1e-9)) > 2):
            notes.append(f"slice {k}: poor subcutaneous ellipse fit, initial guess used")
            ig = oig.copy()
            fitted, _ = fit_and_rasterise(rpfatmaskedge[:, :, k], ig,
                                          radmask, angmask, xmask, ymask, pxsize, fit_rule)
        pfatedgepara[k] = ig

        # 'filling the edge gaps' from the fitted ellipse
        er, _, ep = polar_outer_edge(fitted, radmask, angmask)
        gap = ~present
        pfmaskedger[k, gap] = er[gap]

    # refined body mask from the subcutaneous outer edge
    pfmedgemask = np.zeros((nx, ny, nz), np.uint8)
    for k in range(nz):
        tmp = mask_from_radial_profile(pfmaskedger[k], radmask, angmask)
        tmp = opening(tmp, struo)
        tmp2 = tmp.copy()
        tmp = closing(tmp, struc)
        pfmedgemask[:, :, k] = tmp | tmp2

    return pfmedgemask, pfmaskedger, pfatedgepara


def _visceral_pass(images, fatmask, cfatmask, pfmedgemask, pfatedgepara,
                   pxsize, pysize, radmask, angmask, xmask, ymask, notes, verbose,
                   abd_up=None, debug=None, fit_rule='strict', jump_test=True):
    """Everything from 'get rough nonfat mask' to the visceral/peripheral split.
    jump_test=False reproduces the .pro, whose visceral pass has no fit sanity check."""
    nx, ny, nz = fatmask.shape

    pfmedgemdist = np.stack([morph_distance(pfmedgemask[:, :, i]) for i in range(nz)], axis=2)
    maxpfmedist = pfmedgemdist.max()
    rvnonfatmask = (np.clip(pfmedgemask.astype(int) - fatmask.astype(int), 0, None)
                    * (pfmedgemdist > 10)).astype(np.uint8)

    # 'check if abdomen up'
    utmp = ltmp = 0.0
    for i in range(nz):
        ybdcnt = int(round(-pfatedgepara[i, 1] / pysize + np.floor(ny / 2.0)))
        ybdcnt = int(np.clip(ybdcnt, 1, ny - 1))
        t = rvnonfatmask[:, :, i]
        ltmp += t[:, :ybdcnt].sum()
        utmp += t[:, ybdcnt:].sum()
    is_abd_up = 0 if utmp > ltmp else 1
    if debug is not None:   # the two non-fat counts behind the detection (port-corrections table)
        debug['abd_halves'] = (float(ltmp), float(utmp), int(is_abd_up))
    # Oct 2026: the detection compares non-fat on the two halves and is marginal on
    # some series (it can flip with a small change of threshold), while the stored
    # orientation is known from the image header.  abd_up forces it.
    if abd_up is not None and int(abd_up) != is_abd_up:
        notes.append(f"IsAbdUp forced to {int(abd_up)} (detected {is_abd_up}, "
                     f"nonfat halves {ltmp:.0f}/{utmp:.0f})")
        is_abd_up = int(abd_up)
    if verbose:
        print(f"EstFatVolMRIv2: IsAbdUp = {is_abd_up}")

    # 'refine'
    crvnonfatmask = (np.clip(pfmedgemask.astype(int) - fatmask.astype(int), 0, None)
                     * (pfmedgemdist > maxpfmedist * 2 / 3)).astype(np.uint8)
    pstrud = bdisk(2, 5)
    for k in range(nz):
        ybdcnt = int(round(-pfatedgepara[k, 1] / pysize + np.floor(ny / 2.0)))
        xbdcnt = int(round(pfatedgepara[k, 0] / pxsize + np.floor(nx / 2.0)))
        ybdcnt = int(np.clip(ybdcnt, 1, ny - 1))
        xbdcnt = int(np.clip(xbdcnt, 0, nx - 1))
        hw = pfatedgepara[k, 3] / pxsize / 1.5
        vg = pfatedgepara[k, 4] / pysize / 10
        x0 = int(np.clip(xbdcnt - 1 - hw, 0, nx - 1))
        x1 = int(np.clip(xbdcnt + hw, 0, nx - 1))
        yhi = int(np.clip(ybdcnt + vg, 0, ny - 1))
        ylo = int(np.clip(ybdcnt - 1 - vg, 0, ny - 1))

        upper = np.zeros((nx, ny), np.uint8)
        lower = np.zeros((nx, ny), np.uint8)
        upper[x0:x1 + 1, yhi:] = 1
        lower[x0:x1 + 1, :ylo + 1] = 1
        if is_abd_up:
            tmp, tmp2 = lower, upper
        else:
            tmp, tmp2 = upper, lower
        crvnonfatmask[:, :, k] = (crvnonfatmask[:, :, k] * tmp) | (rvnonfatmask[:, :, k] * tmp2)

        # conditional dilation to drop non-fat outside the visceral area
        rvnonfatmask[:, :, k] = conditional_dilate(crvnonfatmask[:, :, k],
                                                   rvnonfatmask[:, :, k], pstrud)

    # visceral non-fat edge -> ellipse fit -> polynomial/spline repair
    rvnfmaskedger = np.zeros((nz, 360))
    rvnfatedgepara = np.zeros((nz, 5))
    ig = np.array([0, 0, 0,
                   nx / 5.0 * pxsize / 0.859375,
                   ny / 7.0 * pxsize / 0.859375], float)

    for k in range(nz):
        edge = sobel_edge(rvnonfatmask[:, :, k])
        r, _, present = polar_outer_edge(edge, radmask, angmask)
        rvnfmaskedger[k] = r

        oig = ig.copy()
        fitted, ig = fit_and_rasterise(edge, ig, radmask, angmask, xmask, ymask, pxsize, fit_rule)
        # same sanity check as the subcutaneous pass: a fit that jumps is discarded
        # and the previous slice's ellipse used instead (this pass had no check, and
        # one diverged fit seeded every slice after it)
        if jump_test and (np.max(np.abs(oig[0:2] - ig[0:2])) > nx / 8 * pxsize
                          or np.max(np.abs(oig[3:5] - ig[3:5]) / np.maximum(oig[3:5], 1e-9)) > 2):
            notes.append(f"slice {k}: poor visceral ellipse fit, previous ellipse used")
            ig = oig.copy()
            fitted, _ = fit_and_rasterise(edge, ig, radmask, angmask, xmask, ymask, pxsize, fit_rule)
        rvnfatedgepara[k] = ig
        er, _, ep = polar_outer_edge(fitted, radmask, angmask)

        # 'filling ellipse gaps' by quadratic interpolation
        if (~ep).any() and ep.any():
            xs = np.flatnonzero(ep)
            us = np.flatnonzero(~ep)
            er[us] = np.interp(us, xs, er[xs])

        # 'eliminate inappropriate edge over abd. part'
        half = slice(0, 180) if is_abd_up else slice(180, 360)
        idx = np.flatnonzero(rvnfmaskedger[k, half] == 0)
        if idx.size:
            gidx = idx if is_abd_up else idx + 180
            rvnfmaskedger[k, gidx] = er[gidx]

        diff = np.abs(rvnfmaskedger[k, half] - er[half])
        sd = diff.std(ddof=1) if diff.size > 1 else 0.0
        idx = np.flatnonzero(diff > sd * 2)
        if idx.size:
            gidx = idx if is_abd_up else idx + 180
            rvnfmaskedger[k, gidx] = er[gidx]

    # 'refine rvnfmaskedger by fitting' — abdominal sector, degree 4
    if is_abd_up:
        i1s, i1e = -125 + 179, -55 + 179
        i2s, i2e = -135 + 179, -125 + 179
        i3s, i3e = -55 + 179, -45 + 179
        i4s, i4e = -150 + 179, -135 + 179
        i5s, i5e = -45 + 179, -30 + 179
    else:
        i1s, i1e = 55 + 179, 125 + 179
        i2s, i2e = 125 + 179, 135 + 179
        i3s, i3e = 45 + 179, 55 + 179
        i4s, i4e = 135 + 179, 150 + 179
        i5s, i5e = 30 + 179, 45 + 179

    for k in range(nz):
        if is_abd_up:
            thi = np.arange(181) - 180
            idx = np.arange(181)
            off = 0
        else:
            thi = np.arange(181)
            idx = thi + 179
            off = 179
        idx = np.clip(idx, 0, 359)
        coef = np.polyfit(thi, rvnfmaskedger[k, idx], 4)
        abdcfit = np.polyval(coef, thi)

        for a, b in ((i1s, i1e), (i2s, i2e), (i3s, i3e)):
            a2, b2 = np.clip([a, b], 0, 359)
            rvnfmaskedger[k, a2:b2 + 1] = abdcfit[np.clip(np.arange(a2, b2 + 1) - off, 0, 180)]

        diff = np.abs(rvnfmaskedger[k, idx] - abdcfit)
        sd = diff.std(ddof=1) if diff.size > 1 else 0.0
        bad = np.flatnonzero(diff > sd * 2)
        for b_ in bad:
            if (i4s <= b_ < i4e) or (i5s < b_ <= i5e):
                gi = int(np.clip(b_, 0, 359))
                rvnfmaskedger[k, gi] = abdcfit[int(np.clip(gi - off, 0, 180))]

        # 'refine rvnfmaskedger flank parts by fitting' — cubic spline
        if is_abd_up:
            spans = ((15 + 179, 60 + 179), (120 + 179, 165 + 179))
        else:
            spans = ((-60 + 179, -15 + 179), (-165 + 179, -120 + 179))
        for lo, hi in spans:
            lo, hi = int(np.clip(lo, 0, 359)), int(np.clip(hi, 0, 359))
            seg = rvnfmaskedger[k, lo:hi + 1]
            xs = np.flatnonzero(seg != 0)
            us = np.flatnonzero(seg == 0)
            if us.size and xs.size >= 2:
                seg[us] = np.interp(us, xs, seg[xs])
                rvnfmaskedger[k, lo:hi + 1] = seg

    # 'refine a visceral mask from fitted edge'
    rvnfmedgemask = np.zeros((nx, ny, nz), np.uint8)
    for k in range(nz):
        tmp = mask_from_radial_profile(rvnfmaskedger[k], radmask, angmask)
        vabavg = float(np.clip(np.sqrt(max(rvnfatedgepara[k, 3] * rvnfatedgepara[k, 4], 1e-9)), 1.0, nx))
        vstruo = bdisk(2, 4)
        vstruc = bdisk(max(round(vabavg / 2), 1), max(round(vabavg / 1.0), 1))
        tmp = opening(tmp, vstruo)
        tmp2 = tmp.copy()
        tmp = closing(tmp, vstruc)
        rvnfmedgemask[:, :, k] = tmp | tmp2

    revmdist = np.stack([morph_distance(rvnfmedgemask[:, :, i]) for i in range(nz)], axis=2)

    # 'get rough visceral mask using open and close'
    rvisceromask = rvnonfatmask.copy()
    vstruo = bdisk(4, 9)
    for i in range(nz):
        vabavg = float(np.clip(np.sqrt(max(rvnfatedgepara[i, 3] * rvnfatedgepara[i, 4], 1e-9)), 1.0, nx))
        vstruc = bdisk(max(round(vabavg / 6.0), 1), max(round(vabavg / 3.0), 1))
        front = (angmask > 0) & (angmask <= 180)
        back = (angmask <= 0) & (angmask > -180)
        if is_abd_up:
            tmp = (rvnonfatmask[:, :, i].astype(bool)
                   | ((revmdist[:, :, i] > 20) & front)
                   | ((revmdist[:, :, i] > 0) & back))
        else:
            tmp = (rvnonfatmask[:, :, i].astype(bool)
                   | ((revmdist[:, :, i] > 20) & back)
                   | ((revmdist[:, :, i] > 0) & front))
        tmp = opening(tmp.astype(np.uint8), vstruo)
        tmp2 = tmp.copy()
        tmp = closing(tmp, vstruc)
        rvisceromask[:, :, i] = tmp | tmp2

    rvmdist = np.stack([morph_distance(rvisceromask[:, :, i]) for i in range(nz)], axis=2)

    # 'find visceral non-fat edge'
    vnf1 = ((rvmdist <= 32) & rvnonfatmask.astype(bool) & (rvmdist != 0)).astype(np.uint8)
    vnf2 = ((rvmdist <= 33) & (rvmdist > 1)).astype(np.uint8)

    ybdcnt = int(round(-pfatedgepara[-1, 1] / pysize + np.floor(ny / 2.0)))
    ybdcnt = int(np.clip(ybdcnt, 0, ny - 1))
    half = np.zeros((nx, ny, nz), np.uint8)
    half[:, ybdcnt:, :] = 1
    if is_abd_up:
        vnonfatmaskedge = vnf2 * half + vnf1 * (1 - half)
    else:
        vnonfatmaskedge = vnf1 * half + vnf2 * (1 - half)

    # 'filling the gap'
    if is_abd_up:
        rngs = (range(-30, 0), range(-179, -149))
    else:
        rngs = (range(1, 31), range(150, 181))
    ang_r = np.round(angmask).astype(int)
    for k in range(nz):
        tmp = vnonfatmaskedge[:, :, k] * ang_r
        tmp4 = vnf2[:, :, k] * ang_r
        tmp2 = vnonfatmaskedge[:, :, k].copy()
        tmp3 = vnf2[:, :, k]
        for rng in rngs:
            for a in rng:
                sel = (tmp == a)
                if sel.sum() and sel.sum() < 64:
                    sel4 = (tmp4 == a)
                    tmp2[sel4] = tmp3[sel4]
        vnonfatmaskedge[:, :, k] = tmp2

    # 'find peripheral fat mask'
    crvisceromask = (rvnfmedgemask * (revmdist > 10)).astype(np.uint8)
    # NB: cast to int before subtracting. These masks are uint8, and 0 - 1
    # wraps to 255 rather than clipping to 0, which silently turned the
    # peripheral fat mask into most of the image. IDL's BYTE arithmetic
    # saturates instead of wrapping, so the original was safe.
    pfatmask = np.clip(fatmask.astype(np.int16) * (1 - crvisceromask.astype(np.int16))
                       - vnonfatmaskedge.astype(np.int16), 0, None).astype(np.uint8)

    # conditional dilation to remove small visceral fat islands
    cpfatmask = (cfatmask * (revmdist == 0)).astype(np.uint8)
    pfatmask = conditional_dilate(cpfatmask, pfatmask, bdisk(2, 4))

    vfatmask = (fatmask * (1 - pfatmask) * (rvmdist > 0)).astype(np.uint8)
    vzone = (rvmdist > 0).astype(np.uint8)
    if debug is not None:
        # Oct 2026: intermediate masks for inspecting the visceral boundary
        debug.update(rvnonfatmask=rvnonfatmask.copy(),        # non-fat after the conditional dilation
                     rvnfmedgemask=rvnfmedgemask.copy(),      # mask from the fitted/repaired visceral edge
                     rvisceromask=rvisceromask.copy(),        # the visceral zone (vzone = this > 0)
                     crvisceromask=crvisceromask.copy(),      # core removed from SAT before dilation
                     pfmedgemask=pfmedgemask.copy(),          # body from the subcutaneous outer edge
                     front=((angmask > 0) & (angmask <= 180)), # the 'front' sector as the code sees it
                     rvnfatedgepara=rvnfatedgepara.copy(), is_abd_up=is_abd_up)
    return pfatmask, vfatmask, is_abd_up, vzone


def est_fat_vol(images, pxsize, pysize, sspacing, *,
                fat_loc: int = VISCERAL,
                thresh_method: int = 1,
                nobins: int = 100,
                is_vol_thresh: bool = False,
                is_arm_in: int = 0,
                idl_integer_division: bool = True,
                thresh_factor: float = 1.0,
                threshold_rule: str = 'rc',
                abd_up: Optional[int] = None,
                fatmask_in: Optional[np.ndarray] = None,
                cfatmask_in: Optional[np.ndarray] = None,
                bdmask_in: Optional[np.ndarray] = None,
                debug: Optional[dict] = None,
                fit_rule: str = 'strict',
                visceral_jump_test: bool = True,
                thresh_mask_slice: Optional[int] = None,
                verbose: bool = False) -> FatResult:
    """Estimate fat volume (ml) from an axial MRI volume.

    Parameters mirror the IDL function.

    images      : (nx, ny) or (nx, ny, nz) array
    pxsize      : pixel size X (mm)
    pysize      : pixel size Y (mm)
    sspacing    : slice spacing (mm)
    fat_loc     : 0 subcutaneous, 1 visceral (default), 2 total
    thresh_method : 0 lowest-point, 1 centre-of-weight/ISODATA (default)
    is_vol_thresh : threshold over the whole volume rather than per slice
    thresh_factor : multiplier on the threshold (1.0 = .pro as ported and validated;
                    diagnostics only - see _fat_mask)
    threshold_rule : 'rc' (default) plain Ridler-Calvard on the unit-bin masked histogram - the
                    rule the 2006 paper describes; 'idl' the .pro's loop with IDL's
                    integer-histogram bin width; 'float' the same loop on a float-width
                    histogram
    is_arm_in   : arms present in the axial field of view
    fatmask_in  : (Oct 2026) a fat mask supplied from outside, (nx, ny, nz) bool/uint8.
                  When given, the threshold step is skipped and only the SEPARATION runs
                  on this mask - for testing the SAT/VAT split on data where the method's
                  own threshold cannot work (e.g. windowed CT JPEGs, where fat, air and
                  bowel gas share grey 0).  The images are then not used for fat at all.
    cfatmask_in : the 'confident fat' seed that the 1.3x threshold mask normally provides
                  (it seeds the conditional dilation that grows subcutaneous fat);
                  defaults to fatmask_in when not given.
    bdmask_in   : a body mask supplied from outside; the method's own body mask assumes
                  bright fat (MRI) and is used only when this is not given.
    debug       : optional dict; if given, the visceral pass stores its intermediate masks
                  in it (see the end of _visceral_pass).
    The next three exist (Oct 2026) to measure the port corrections; their defaults are
    the corrected, validated method and leave every earlier result unchanged.
    fit_rule    : 'strict' (default) accept only converged, in-field ellipse fits; 'fov' any
                  in-field fit; 'any' whatever the optimiser returns (the port before the
                  17 Sep 2026 fix).  See ellipse.FIT_RULES; counts in ellipse.FIT_STATS.
    visceral_jump_test : the fit sanity check in the visceral pass (default True); False is
                  the .pro, which checks only the subcutaneous fit.
    thresh_mask_slice : None (default) or a slice index k: every slice's threshold is taken
                  inside slice k's body outline, as the 2003/2008 threshold program did with
                  k = 0 (see _fat_mask).
    idl_integer_division : reproduce the original's integer division in the
                  ISODATA threshold. Leave True to match historical results.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[:, :, None]
    if images.ndim != 3:
        raise ValueError("Images must be either one 2D image or a 3D volume.")
    nx, ny, nz = images.shape
    notes: list = []

    radmask, angmask, xmask, ymask, imask, jmask = coordinate_masks(nx, ny, pxsize, pysize)

    if bdmask_in is not None:
        bdmask = (np.asarray(bdmask_in).reshape(nx, ny, nz) > 0).astype(np.uint8)
    else:
        bdmask, _ = _body_mask(images, pxsize, pysize, radmask, angmask, imask, jmask,
                               is_arm_in, verbose)
    if fatmask_in is not None:
        fatmask = ((np.asarray(fatmask_in).reshape(nx, ny, nz) > 0) & (bdmask > 0)).astype(np.uint8)
        cfatmask = (fatmask if cfatmask_in is None else
                    ((np.asarray(cfatmask_in).reshape(nx, ny, nz) > 0) & (fatmask > 0)).astype(np.uint8))
        thregray = np.full(nz, np.nan)
        notes.append("fat mask supplied (fatmask_in): threshold step skipped")
    else:
        fatmask, cfatmask, thregray = _fat_mask(images, bdmask, thresh_method, nobins,
                                                is_vol_thresh, idl_integer_division, verbose,
                                                thresh_factor, threshold_rule, thresh_mask_slice)

    voxel_ml = pxsize * pysize * sspacing / 1000.0
    fatvol = voxel_ml * float(fatmask.sum())

    fmask, vfmask, vmaskedge, is_abd_up, zones = fatmask, None, None, None, None

    if fat_loc != TOTAL:
        pfmedgemask, pfmaskedger, pfatedgepara = _subcutaneous_ellipse_pass(
            fatmask, bdmask, pxsize, pysize, radmask, angmask, xmask, ymask, notes,
            fit_rule=fit_rule)
        pfatmask, vfatmask, is_abd_up, vzone = _visceral_pass(
            images, fatmask, cfatmask, pfmedgemask, pfatedgepara,
            pxsize, pysize, radmask, angmask, xmask, ymask, notes, verbose,
            abd_up=abd_up, debug=debug, fit_rule=fit_rule, jump_test=visceral_jump_test)
        body = bdmask.astype(bool)
        zones = body.astype(np.uint8)
        zones[body & vzone.astype(bool)] = 2
        n_out = int((vzone.astype(bool) & ~body).sum())
        if n_out:
            notes.append(f"visceral zone: {n_out} voxels outside the body mask (left as 0)")

        if fat_loc == SUBCUTANEOUS:
            fatvol = voxel_ml * float(pfatmask.sum())
            fmask, vfmask = pfatmask, vfatmask
        else:
            fatvol = voxel_ml * float(vfatmask.sum())
            fmask = vfatmask

    maskedge = np.stack([sobel_edge(fmask[:, :, i]) for i in range(nz)], axis=2)
    if fat_loc == SUBCUTANEOUS and vfmask is not None:
        vmaskedge = np.stack([sobel_edge(vfmask[:, :, i]) for i in range(nz)], axis=2)

    return FatResult(volume_ml=fatvol, fat_loc=fat_loc, fmask=fmask,
                     maskedge=maskedge, vfmask=vfmask, vmaskedge=vmaskedge,
                     is_abd_up=is_abd_up, bdmask=bdmask, fatmask=fatmask,
                     thresholds=thregray, zones=zones, notes=notes)
