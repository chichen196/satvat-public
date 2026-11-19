"""wallsep.py - SAT/VAT separation by following the abdominal wall (Oct 2026).

A replacement for the 2006 ellipse boundary.  The anatomical SAT/VAT boundary is
the muscle wall: a closed non-fat ring (rectus, obliques, transversus; paraspinal,
quadratus lumborum and psoas with the spine at the back).  Subcutaneous fat is the
fat OUTSIDE that ring, visceral fat the fat INSIDE it - including the fat lying
between the wall and the organs, which the ellipse approach loses when it is
plentiful.  No polar fitting of a shape, so concave outlines are fine.

wall_zone(F, B, px_mm, ...) -> (zone, info)

    F      fat mask (rows, cols) bool          - the pixels to be split
    B      body mask (rows, cols) bool, filled - everything inside the skin
    nonfat optional (rows, cols) bool          - what counts as "wall material";
           default B & ~F.  Pass the full non-fat mask (all fat excluded, not
           only the fat to be split) when F is a subset of the fat.
    zone   bool (rows, cols): True inside the wall.  VAT = F & zone, SAT = F & ~zone.

    nonfat_high (optional, Oct 2026, for MRI): a second, more inclusive wall material -
           everything below a higher intensity cut (otsu3(): the upper threshold of a
           three-class Otsu split of the trunk, dark | muscle and organs | fat).  Used
           only when the ring built from `nonfat` is OPEN: fat outside the zone lying
           deeper than the wall surface of its own sector (open_margin_mm below it) is
           the leaked cavity; when it exceeds open_frac of the fat (and open_min_mm2)
           a second pass adds the nonfat_high pixels deeper than the expected wall depth
           (the sector profile of pass 1, interpolated across sectors without wall,
           minus admit_margin_mm) and closes again, escalating the closing radius
           (escalate_mm) if the ring is still open.  Why: in obese subjects at 9 mm
           slices the thin anterolateral wall is partial-volumed with fat on both sides
           and lies ABOVE the fat threshold (0033: muscle p90 323 vs fat p10 322), so it
           is not in B & ~F and the cavity empties into the SAT (VAT->SAT 22-56% in 4 of
           40 CHAOS cases).  The higher cut alone would also turn dim SAT into wall and
           close the ring out in the SAT (SAT->VAT 9-23%), hence the depth restriction
           and the use only where pass 1 fails.  Result on those four cases: VAT->SAT
           3-5%, SAT->VAT 1.5-4%; cases whose ring closes in pass 1 are untouched.
    skin_max_mm (optional, Oct 2026, for MRI): take the skin layer from the image
           instead of a fixed depth.  The non-fat that is contiguous with the outside of
           the body (dermis, the body mask's partial-volume halo, the ramp into the fat)
           is removed from the wall material down to skin_max_mm, whatever its depth.
           On T1 MRI that band is 4-8 mm (low-signal dermis, mask edge in the halo),
           thicker than a fixed 3 mm, and it survived the opening as a "wall" at the skin.
           Where there is no fat between skin and muscle the band includes muscle, so the
           cap limits the loss to skin_max_mm of its depth.  skin_mm stays as a floor.
    anterior, open_post_mm, skin_post_mm, post_from (optional, Oct 2026): behind the
           body centre the wall is thick muscle, while the SAT there often holds a
           well-formed membranous layer and the dermis of the back is thick.  Given
           which way the patient faces (anterior='up'|'down'|'left'|'right'), the
           posterior part of the body gets its own opening radius and skin depth.

Settings used (Oct 2026):
    CT, AATTCT-IDS (0.78 mm px):  open_mm=0.78 close_mm=9.4 skin_mm=3 open_post_mm=3.1,
        leak repair on  (the manuscript's CT runs; open/close values reproduce the runs
        made before the _disk rounding fix, see _disk).
    MRI, CHAOS T1 in-phase (1.0-1.9 mm px): the same plus skin_max_mm=10, close_mm=25,
        close_steps_mm=(), leak=False, and nonfat_high = B & (image < otsu3(image[B])[1])
        for the second pass (see nonfat_high).  Two things differ on MRI.  (a) The non-fat band
        at the trunk edge is 4-8 mm deep (body-mask edge in the dark halo, low-signal
        dermis, the ramp into fat), so with a fixed 3 mm skin layer a 5 mm band survived
        the opening as a "wall" at the skin and the ring closed there (0030: all SAT
        became VAT).  skin_max_mm takes the skin from the image instead.  (b) The gaps in
        the ring are tangential - at the posterolateral corner (lumbar triangle) the
        obliques end and the paraspinals begin with only aponeurosis between, 2-4 cm at
        these pixel sizes - which a 9 mm closing cannot bridge and the radial depth
        profile cannot see.  The leak repair then fires on thick SAT instead (a 5 cm
        flank pad next to 1.5 cm neighbours at the iliac crest, 0001) and pushes it
        inside; in both test cases every bin it flagged was SAT by the reference model.
        A 25 mm closing seals the corner gaps on its own, and the posterolateral pockets
        deep to latissimus mostly stay outside (their mouths are wider than that).
        Fat misassigned against the model, within the threshold fat: 0001 SAT->VAT 0.9%,
        VAT->SAT 2.0%; 0030 3.3% / 0.4% (vs 11% / 1.5% and 3.3% / 0.9% with the CT
        setting plus the skin band, and 48% SAT->VAT without the skin band).

Steps:
 1. N = nonfat inside the body, minus a thin skin layer (depth <= skin_mm), so the
    dermis cannot act as wall.
 2. N = opening(N, open_mm): removes membranes (Scarpa's fascia, the back's
    membranous layer, septa) and partial-volume lines - layers thinner than about
    2 x open_mm; then drop components smaller than min_area_mm2 (vessels in the
    SAT).  Muscle (>= 5 mm thick) survives.  A layer that survives here becomes
    wall, and the fat between it and the muscles is then counted as visceral.
 3. Close the ring and fill it: W = closing(N, R), M = fill_holes(W).  R starts at
    close_mm and is raised once (close_steps_mm) if the cavity did not fill - a
    closed ring fills to at least closed_frac of the body.  The closing bridges
    small gaps in the ring (linea alba, aponeuroses lost to partial volume).
 4. Leak repair for wider gaps, whether the closing bridged them wrongly (along an
    organ behind the gap) or not at all.  The wall's outer-surface depth per
    angular bin about the body centroid (the shallowest wall pixel) is a smooth
    profile; a bin whose wall is far deeper than its neighbours, or has no wall,
    is flagged and its depth interpolated from the neighbours.  Pixels deeper
    than the (interpolated) wall surface of their bin are "deep"; the deep fat
    connected to a flagged bin is the leaked cavity and goes back into the zone.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _disk(r):
    """Disk of radius r pixels, ROUNDED to the nearest pixel (min 1).

    Until 14 Oct 2026 the radius was truncated (int), so open_mm=1.5 at 0.78 mm/px
    (1.92 px) gave a radius-1 plus-shaped element: the opening then removed only
    layers up to 2 px thick and a 3-4 px fascia or dermis survived as wall.
    To reproduce runs made before the fix: open_mm=0.78 and close_mm=9.4 at px=0.78.
    """
    r = max(int(round(r)), 1)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _closing(N, r_px):
    """Closing of N by a Euclidean disk of radius r_px pixels, outside the array = background
    (the same result as ndi.binary_closing(N, _disk(r), border_value=0), checked pixel for
    pixel, but by two distance transforms, so a 25 mm radius costs no more than a 10 mm one)."""
    r = float(max(int(round(r_px)), 1))
    p = int(r) + 1
    N = np.asarray(N, bool)
    D = ndi.distance_transform_edt(~N) <= r               # dilation (cut at the array edge,
    Dp = np.pad(D, p)                                      #  as binary_closing does)
    E = ndi.distance_transform_edt(Dp) > r                 # erosion, outside = background
    return E[p:-p, p:-p]


def otsu3(values, bins=256):
    """Three-class Otsu thresholds (t_low, t_high) of a 1-D sample: maximises the
    between-class variance over all pairs of cut points on a `bins`-bin histogram
    (same result as skimage's threshold_multiotsu(classes=3))."""
    vals = np.asarray(values, float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return (np.nan, np.nan)
    h, e = np.histogram(vals, bins)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h).astype(float); m = np.cumsum(h * c)
    W, M = w[-1], m[-1]; mu = M / W
    i = np.arange(1, bins - 1)[:, None]; j = np.arange(2, bins)[None, :]       # cuts after bin i-1 and j-1
    ok = j > i
    w0 = w[i - 1]; m0 = m[i - 1]
    w1 = w[j - 1] - w0; m1 = m[j - 1] - m0
    w2 = W - w[j - 1]; m2 = M - m[j - 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (w0 * (m0 / w0 - mu) ** 2 + w1 * (m1 / w1 - mu) ** 2 + w2 * (m2 / w2 - mu) ** 2)
    v = np.where(ok & (w0 > 0) & (w1 > 0) & (w2 > 0), v, -np.inf)
    k = np.unravel_index(np.argmax(v), v.shape)
    return float(c[i[k[0], 0] - 1]), float(c[j[0, k[1]] - 1])


def _sectors(B, depth, N, bin_deg=3):
    """Angular sectors about the body centroid: a function pixel->sector and, per sector,
    the depth of the shallowest pixel of N (inf where N has none)."""
    cy, cx = ndi.center_of_mass(B)
    nb = int(round(360 / bin_deg))
    ang = lambda y, x: np.minimum(((np.degrees(np.arctan2(y - cy, x - cx)) % 360.0) / bin_deg).astype(int), nb - 1)
    t_wall = np.full(nb, np.inf)
    yy, xx = np.nonzero(N)
    if yy.size:
        np.minimum.at(t_wall, ang(yy, xx), depth[yy, xx])
    return ang, t_wall


def _circular_interp(vals, ok):
    """Linear interpolation of a circular profile over the bins where ok is False."""
    n = len(vals)
    out = vals.astype(float).copy()
    idx = np.flatnonzero(ok)
    if idx.size == 0 or idx.size == n:
        return out
    xs = np.concatenate([idx, [idx[0] + n]])
    ys = np.concatenate([vals[idx], [vals[idx[0]]]])
    bad = np.flatnonzero(~ok)
    q = np.where(bad < idx[0], bad + n, bad)
    out[bad] = np.interp(q, xs, ys)
    return out


def _posterior(B, anterior, post_from=0.0):
    """Pixels of the body behind its centre.  anterior = side of the image the
    patient's front faces: 'up' (low row index), 'down', 'left' or 'right'.
    post_from shifts the dividing line backwards by that fraction of the body's
    half-depth (0 = through the centroid)."""
    if anterior not in ('up', 'down', 'left', 'right'):
        raise ValueError("anterior must be 'up', 'down', 'left' or 'right' when a posterior setting is used")
    rows, cols = np.nonzero(B)
    yy, xx = np.indices(B.shape)
    if anterior in ('up', 'down'):
        c, lo, hi, coord = rows.mean(), rows.min(), rows.max(), yy
    else:
        c, lo, hi, coord = cols.mean(), cols.min(), cols.max(), xx
    if anterior in ('up', 'left'):
        return B & (coord > c + post_from * (hi - c))
    return B & (coord < c - post_from * (c - lo))


def skin_band(nonfat, B, px_mm=0.8, *, skin_max_mm=10.0, seed_mm=None, depth=None):
    """Non-fat contiguous with the outside of the body, down to skin_max_mm.

    nonfat, B  bool (rows, cols); B filled.  depth: EDT of B in pixels (computed if None).
    seed_mm    the outermost layer that starts the band (default 2 px).
    Returns the band (bool): the connected pieces of nonfat & (depth <= skin_max_mm) that
    reach the outermost layer.  Fat separates the band from the muscle; a membrane or a
    spot where fat is absent lets it reach muscle, but only to skin_max_mm deep."""
    B = np.asarray(B, bool)
    px = float(px_mm)
    if depth is None:
        depth = ndi.distance_transform_edt(B)
    M = np.asarray(nonfat, bool) & B & (depth <= skin_max_mm / px)
    seed = M & (depth <= (2.0 if seed_mm is None else seed_mm / px))
    lab, n = ndi.label(M)
    if not n or not seed.any():
        return np.zeros_like(B)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seed])] = True
    keep[0] = False
    return keep[lab]


def wall_zone(F, B, px_mm=0.8, *, nonfat=None, nonfat_high=None, open_margin_mm=5.0, open_frac=0.02,
              open_min_mm2=300.0, admit_margin_mm=10.0, escalate_mm=(40.0,), return_steps=False, **kw):
    """See the module docstring.  Without nonfat_high this is the single pass (_wall_pass);
    with it, the ring is tested for leaks and rebuilt with the brighter wall material where
    it is open."""
    F = np.asarray(F, bool)
    B = np.asarray(B, bool)
    if F.shape != B.shape or F.ndim != 2:
        raise ValueError("F and B must be the same 2-D shape")
    px = float(px_mm)
    out = _wall_pass(F, B, px, nonfat=nonfat, return_steps=True, **kw)
    zone, info, st = out
    info['pass'] = 1
    info['open_px'] = 0
    if nonfat_high is None or not B.any() or not st.get('N', B).any():
        return (zone, info, st) if return_steps else (zone, info)

    depth = ndi.distance_transform_edt(B)
    ang, t_wall = _sectors(B, depth, st['N'])
    has = np.isfinite(t_wall)
    if not has.any():
        return (zone, info, st) if return_steps else (zone, info)
    t_fill = _circular_interp(np.where(has, t_wall, 0.0), has)

    def open_px(z):
        py, pxl = np.nonzero(F & ~z)
        if not py.size:
            return 0
        return int((depth[py, pxl] > t_fill[ang(py, pxl)] + open_margin_mm / px).sum())

    op = open_px(zone)
    info['open_px'] = op
    thr = max(open_frac * F.sum(), open_min_mm2 / (px * px))
    if op <= thr:
        return (zone, info, st) if return_steps else (zone, info)

    # pass 2: the brighter wall material, but only deeper than the expected wall depth
    N_low = (B & ~F) if nonfat is None else (np.asarray(nonfat, bool) & B)
    yy, xx = np.nonzero(np.asarray(nonfat_high, bool) & B)
    deep = depth[yy, xx] > t_fill[ang(yy, xx)] - admit_margin_mm / px
    N2 = N_low.copy()
    N2[yy[deep], xx[deep]] = True
    info['pass'] = 2
    info['high_added_px'] = int((N2 & ~N_low).sum())
    best = None
    for r_mm in (kw.get('close_mm', 10.0),) + tuple(escalate_mm):
        z2, i2, st2 = _wall_pass(F, B, px, nonfat=N2, return_steps=True, **dict(kw, close_mm=r_mm))
        op2 = open_px(z2)
        if best is None or op2 < best[0]:
            best = (op2, z2, i2, st2, r_mm)
        if op2 <= thr:
            break
    op2, zone, info2, st, r_mm = best
    info2.update(info, open_px2=op2, close2_mm=r_mm)
    return (zone, info2, st) if return_steps else (zone, info2)


def _wall_pass(F, B, px_mm=0.8, *, nonfat=None, skin_mm=3.0, skin_max_mm=None, open_mm=1.5,
               anterior=None, open_post_mm=None, skin_post_mm=None, post_from=0.0,
               min_area_mm2=150.0, close_mm=10.0, close_steps_mm=(15.0,),
               closed_frac=0.005, leak=True, leak_ratio=1.6, leak_margin_mm=3.0,
               cut_margin_mm=4.0, gap_margin_mm=1.0, bin_deg=3, window_deg=90, ref_pct=25, return_steps=False):
    F = np.asarray(F, bool)
    B = np.asarray(B, bool)
    info = {}
    px = float(px_mm)
    if not B.any():
        z = np.zeros_like(B)
        return (z, info, {}) if return_steps else (z, info)

    depth = ndi.distance_transform_edt(B)                      # pixels from the skin
    N = (B & ~F) if nonfat is None else (np.asarray(nonfat, bool) & B)
    post = _posterior(B, anterior, post_from) if (open_post_mm is not None or skin_post_mm is not None) else None
    skin = np.full(B.shape, skin_mm / px)
    if post is not None and skin_post_mm is not None:
        skin[post] = skin_post_mm / px
    N = N & (depth > skin)
    info['skin_band_px'] = 0
    if skin_max_mm is not None:
        band = skin_band((B & ~F) if nonfat is None else (np.asarray(nonfat, bool) & B), B, px,
                         skin_max_mm=skin_max_mm, depth=depth)
        N = N & ~band
        info['skin_band_px'] = int(band.sum())
    N0 = N.copy()                                              # wall material before the opening

    # 2. membranes and small blobs out (optionally a stronger opening behind the body
    #    centre, where the true wall - paraspinal, quadratus lumborum, psoas - is thick
    #    and the back's membranous layer and thick dermis are what must not become wall)
    Na = ndi.binary_opening(N, structure=_disk(open_mm / px))
    if post is not None and open_post_mm is not None:
        Np = ndi.binary_opening(N, structure=_disk(open_post_mm / px))
        N = np.where(post, Np, Na)
    else:
        N = Na
    info['n_post_px'] = int(post.sum()) if post is not None else 0
    lab, n = ndi.label(N)
    if n:
        area = np.bincount(lab.ravel())
        keep = area >= (min_area_mm2 / (px * px))
        keep[0] = False
        N = keep[lab]
    info['n_wall_components'] = int(n)

    # 3. close the ring and fill it, raising the radius until the cavity fills
    body_px = int(B.sum())
    W = M = None
    for r_mm in (close_mm,) + tuple(r for r in close_steps_mm if r > close_mm):
        W = _closing(N, r_mm / px)
        M = ndi.binary_fill_holes(W) & B
        filled = int((M & ~W).sum())
        info['close_mm'] = float(r_mm)
        info['filled_px'] = filled
        if filled >= closed_frac * body_px:
            break
    info['closed'] = bool(info['filled_px'] >= closed_frac * body_px)
    zone = M.copy()

    # 4. leak repair from the wall-depth profile
    info['leak_bins'] = 0
    info['leak_moved_px'] = 0
    if leak and N.any():
        cy, cx = ndi.center_of_mass(B)
        nb = int(round(360 / bin_deg))
        ang = lambda y, x: np.minimum(((np.degrees(np.arctan2(y - cy, x - cx)) % 360.0) / bin_deg).astype(int), nb - 1)
        yy, xx = np.nonzero(N)
        b = ang(yy, xx)
        t_wall = np.full(nb, np.inf)
        np.minimum.at(t_wall, b, depth[yy, xx])                 # shallowest wall pixel per bin
        has = np.isfinite(t_wall)
        half = max(int(round(window_deg / bin_deg)) // 2, 1)
        ref = np.full(nb, np.nan)
        for i in range(nb):
            nbr = [(i + k) % nb for k in range(-half, half + 1) if k]
            v = t_wall[nbr][has[nbr]]
            if v.size:
                ref[i] = np.percentile(v, ref_pct)      # a low quantile: true wall wins over a gap
        margin = leak_margin_mm / px
        glob = np.median(t_wall[has]) if has.any() else 0.0
        flagged = ~has | (np.isfinite(ref) & (t_wall > leak_ratio * ref + margin)) \
            | (has & (t_wall > 2.5 * glob + margin))
        if flagged.any() and (~flagged).any():
            t_fill = _circular_interp(np.where(has, t_wall, 0.0), ~flagged)
            # everything outside the zone that lies deeper than the wall surface of its
            # bin is "deep"; the leaked region is the deep fat connected to a flagged bin
            py, pxl = np.nonzero(B & ~zone)
            pb = ang(py, pxl)
            marg = np.where(flagged[pb], gap_margin_mm, cut_margin_mm) / px
            deep_sel = depth[py, pxl] > t_fill[pb] + marg
            deep = np.zeros_like(B); deep[py[deep_sel], pxl[deep_sel]] = True
            seed = np.zeros_like(B); ss = deep_sel & flagged[pb]; seed[py[ss], pxl[ss]] = True
            lab, n = ndi.label(deep)
            if n and seed.any():
                hit = np.zeros(n + 1, bool); hit[np.unique(lab[seed])] = True; hit[0] = False
                L = hit[lab]
                zone |= L
                info['leak_moved_px'] = int((L & F).sum())
            info['leak_bins'] = int(flagged.sum())

    if return_steps:
        return zone, info, dict(N0=N0, N=N, W=W, M=M)
    return zone, info


def wall_split(F, B, px_mm=0.8, **kw):
    """Convenience: returns (sat, vat, zone, info)."""
    zone, info = wall_zone(F, B, px_mm, **kw)
    return F & ~zone, F & zone, zone, info
