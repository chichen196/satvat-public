"""
Faithful port of FIT2DELLIP (C.-H. Chen, 1996-2006)
and its objective function FIT2DELLFUNC.

Fits edge points to

    f(x, y) = [(x-x0)cos(th) - (y-y0)sin(th)]^2 / a^2
            + [-(x-x0)sin(th) + (y-y0)cos(th)]^2 / b^2 - 1

by minimising sum(f^2) with a Nelder-Mead simplex. The 2006 revision replaced
POWELL with AMOEBA "because of much better performance"; scipy's Nelder-Mead is
the same algorithm, and the initial simplex is built the way IDL's AMOEBA builds
it — from P0 and P0 + SCALE[i]*e_i — so the search path is comparable.

THREE THINGS THE EARLIER SUBSTITUTE GOT WRONG, now fixed:

1. `th` is in RADIANS, not degrees. The substitute returned degrees, which were
   then fed straight back into cos/sin.
2. The returned outline is the band between the ellipses with semi-axes
   (a +/- xystep/2), not a tolerance around f = 1. Its thickness is therefore
   set by the pixel size, as intended.
3. The fit is SEEDED from IG and refines it. The substitute used a direct
   least-squares conic fit that ignored IG, losing the slice-to-slice continuity
   the caller depends on — IG is passed by reference in IDL and carries forward
   from one slice to the next.

Only the FINDEDGE=0, REDGE=0 path is exercised by estfatvolmriv2; the
morphological edge-finding branches are ported but lightly tested.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .idlcompat import bdisk, closing, dilate, opening, sobel_magnitude


def _objective(free, fit_mask, ig, xcoor, ycoor):
    """FIT2DELLFUNC: sum over edge points of f(x, y)^2."""
    p = np.asarray(ig, float).copy()
    p[fit_mask] = free
    x0, y0, th, a, b = p
    if a == 0 or b == 0:
        return np.inf
    ct, st = np.cos(th), np.sin(th)
    u = ((xcoor - x0) * ct - (ycoor - y0) * st) / a
    v = (-(xcoor - x0) * st + (ycoor - y0) * ct) / b
    return float(np.sum((u ** 2 + v ** 2 - 1.0) ** 2))


def _amoeba_simplex(p0, scale):
    """IDL AMOEBA's initial simplex: P0, then P0 + SCALE[i] * e_i."""
    n = len(p0)
    simplex = np.tile(np.asarray(p0, float), (n + 1, 1))
    for i in range(n):
        simplex[i + 1, i] += scale[i]
    return simplex


def _edge_coordinates(tedge, xystep):
    """XCoor/YCoor in mm, with the .pro's centring and y-flip."""
    xlen, ylen = tedge.shape
    xcenter, ycenter = xlen // 2, ylen // 2      # floor()
    ii, jj = np.nonzero(np.asarray(tedge) == 1)
    return ((ii - xcenter) * xystep).astype(float), \
           ((ycenter - 1 - jj) * xystep).astype(float)


def _ellipse_band(ig, xlen, ylen, xystep):
    """REDGE=0 return: inside (a + h/2, b + h/2) but outside (a - h/2, b - h/2)."""
    x0, y0, th, a, b = ig
    half = xystep / 2.0
    ao, bo, ai, bi = a + half, b + half, a - half, b - half
    if ao <= 0 or bo <= 0:
        return np.zeros((xlen, ylen), np.uint8)
    xcenter, ycenter = xlen // 2, ylen // 2
    i = np.arange(xlen)[:, None]
    j = np.arange(ylen)[None, :]
    x = (i - xcenter) * xystep * np.ones((1, ylen))
    y = (ycenter - 1 - j) * xystep * np.ones((xlen, 1))
    ct, st = np.cos(th), np.sin(th)
    u = (x - x0) * ct - (y - y0) * st
    v = -(x - x0) * st + (y - y0) * ct
    inside_outer = (u / ao) ** 2 + (v / bo) ** 2 < 1.0
    if ai <= 0 or bi <= 0:
        return inside_outer.astype(np.uint8)
    outside_inner = (u / ai) ** 2 + (v / bi) ** 2 > 1.0
    return (inside_outer & outside_inner).astype(np.uint8)


# Oct 2026, for the port-corrections table: how a fit is accepted.
#   'strict' (default, the 17 Sep 2026 fix): only a converged fit describing an ellipse
#            inside the field of view replaces the seed; an implausible seed is reset to
#            the default guess.  This is what IDL's AMOEBA gives (it returns -1 and the
#            seed stands when NMAX is reached).
#   'fov'    any fit inside the field of view, converged or not.
#   'any'    the port before the fix: whatever the optimiser returns, no seed check
#            (a diverged fit then seeds the next slice).
FIT_RULES = ('strict', 'fov', 'any')
FIT_STATS = {'calls': 0, 'converged': 0, 'accepted': 0, 'seed_reset': 0}


def reset_fit_stats():
    for k in FIT_STATS:
        FIT_STATS[k] = 0


def fit2dellip(edge, ig, thresh: float = 0.4, *,
               tol: float = 1e-8, fit_opt=None, find_edge: int = 2,
               thresh_fe: float | None = None, xystep: float = 2.0,
               redge: int = 0, max_iter: int = 5000, fit_rule: str = 'strict'):
    """Port of FIT2DELLIP. Returns (result_image, fitted_ig).

    In IDL, IG was updated in place; here the fitted parameters are the second
    return value. ig is [x0, y0, th_radians, a, b] in millimetres.
    fit_rule: see FIT_RULES above ('strict' is the validated behaviour).
    """
    if fit_rule not in FIT_RULES:
        raise ValueError(f"fit_rule must be one of {FIT_RULES}")
    edge = np.asarray(edge)
    if edge.ndim != 2:
        raise ValueError("FIT2DELLIP: Edge must be a 2-D array.")
    xlen, ylen = edge.shape
    ig = np.asarray(ig, float).copy()
    fit_opt = np.ones(5, int) if fit_opt is None else np.asarray(fit_opt, int)

    tedge, tmp = None, None
    if find_edge == 2:
        if thresh_fe is None:
            raise ValueError("FIT2DELLIP: The keyword ThreshFE must be set.")
        t = (edge > thresh_fe).astype(np.uint8)
        t = opening(t, bdisk(2, 5))
        t2 = t.copy()
        t = closing(t, bdisk(12, 25))
        t = t | t2
        e = sobel_magnitude(t.astype(float) * 100.0)
        e = (e > e.max() * thresh).astype(np.uint8) if e.max() > 0 else e.astype(np.uint8)
        if redge == 1:
            tedge = e
        else:
            tmp = dilate(e, bdisk(2, 5))

    if find_edge >= 1 and redge != 1:
        te = sobel_magnitude(edge.astype(float) * 100.0)
        if find_edge == 2 and tmp is not None:
            te = te * tmp
        tedge = (te > te.max() * thresh).astype(np.uint8) if te.max() > 0 \
            else np.zeros(edge.shape, np.uint8)

    if find_edge == 0:
        tedge = (edge == 1).astype(np.uint8)

    xcoor, ycoor = _edge_coordinates(tedge, xystep)
    if xcoor.size < 5:
        return (_ellipse_band(ig, xlen, ylen, xystep) if redge == 0 else tedge), ig

    # a seed that could not be an ellipse in this field of view (carried over from
    # an earlier bad slice) is replaced by the driver's default starting guess
    fov = max(xlen, ylen) * xystep
    FIT_STATS['calls'] += 1
    if fit_rule != 'any' and not (np.all(np.isfinite(ig)) and xystep <= ig[3] <= fov
                                  and xystep <= ig[4] <= fov
                                  and abs(ig[0]) <= fov and abs(ig[1]) <= fov):
        ig = np.array([0.0, 0.0, 0.0, xlen / 4.0 * xystep, ylen / 6.0 * xystep])
        FIT_STATS['seed_reset'] += 1

    fit_mask = fit_opt > 0
    free_idx = np.flatnonzero(fit_mask)
    p0 = ig[free_idx]
    scale_all = np.array([xlen / 8.0, ylen / 8.0, np.pi / 4.0, ig[3], ig[4]])
    scale = scale_all[free_idx]
    if not np.all(np.isfinite(scale)) or np.any(scale == 0):
        scale = np.where(scale == 0, 1.0, scale)

    res = minimize(_objective, p0, args=(fit_mask, ig, xcoor, ycoor),
                   method="Nelder-Mead",
                   options={"initial_simplex": _amoeba_simplex(p0, scale),
                            "fatol": tol, "xatol": tol,
                            "maxiter": max_iter, "maxfev": max_iter})

    # IDL's AMOEBA returns -1 (and leaves P0 alone) when it has not converged
    # within NMAX evaluations; scipy hands back wherever the simplex wandered
    # to.  Accepting that point is what let a diverged fit feed axes of ~1e17 mm
    # into bdisk() downstream.  Keep the seed unless the fit both converged and
    # describes an ellipse that could exist inside the field of view.
    accepted = False
    FIT_STATS['converged'] += int(bool(res.success))
    if fit_rule == 'any':
        ig = ig.copy(); ig[free_idx] = res.x; accepted = True
    elif (res.success or fit_rule == 'fov') and np.all(np.isfinite(res.x)):
        cand = ig.copy(); cand[free_idx] = res.x
        cand[3:5] = np.abs(cand[3:5])
        if (xystep <= cand[3] <= fov and xystep <= cand[4] <= fov
                and abs(cand[0]) <= fov and abs(cand[1]) <= fov):
            ig = cand; accepted = True
    ig[3:5] = np.abs(ig[3:5])          # para(3:4) = abs(para(3:4))
    FIT_STATS['accepted'] += int(accepted)
    fit2dellip.last_accepted = accepted

    return (_ellipse_band(ig, xlen, ylen, xystep) if redge == 0 else tedge), ig


def fit_and_rasterise(edge_mask, ig, radmask, angmask, xmask, ymask, xystep, fit_rule='strict'):
    """Adapter for the call sites in segment.py. Equivalent to the .pro's
    `fitted = fit2dellip(edge, IG, FINDEDGE=0, XYSTEP=pxsize, REDGE=0)`."""
    return fit2dellip(edge_mask, ig, find_edge=0, xystep=xystep, redge=0, fit_rule=fit_rule)
