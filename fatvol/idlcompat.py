"""
Small helpers that reproduce the IDL primitives used by ESTFATVOLMRIV2.

Array convention
----------------
Volumes keep the ORIGINAL IDL axis order ``(nx, ny, nz)`` — i.e. IDL's
``images[i, j, k]`` is Python's ``images[i, j, k]``. This is deliberate: it makes
the port line-comparable with the .pro source, so that the two can be checked
against each other function by function. It is not the usual numpy
convention, so the DICOM reader transposes into this order on load.

IDL's row 0 is displayed at the BOTTOM, and the original code encodes that in
``y = (ycenter - 1 - j) * pysize``. That sign convention is preserved here because
IsAbdUp (abdomen-up detection) depends on it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# structuring elements
# --------------------------------------------------------------------------
def bdisk(r: float, sidelen: int) -> np.ndarray:
    """Binary disk of radius `r` in a `sidelen` x `sidelen` array.

    Faithful port of bdisk.pro. Note the centre is ``floor(sidelen/2)``, NOT
    ``(sidelen - 1)/2``: for EVEN sidelen the disk sits half a pixel off-centre
    towards higher indices. That asymmetry is real and is preserved here —
    bdisk(2, 4) and bdisk(round(v/2), round(v)) are both used with even sides.
    """
    sidelen = max(int(round(sidelen)), 1)
    r = float(r)
    c = sidelen // 2                      # floor(sidelen/2), as in the .pro
    i = np.arange(sidelen)[:, None]
    j = np.arange(sidelen)[None, :]
    x = (i - c).astype(float)
    y = (j - c).astype(float)
    return ((x ** 2 + y ** 2) <= r ** 2).astype(np.uint8)


# --------------------------------------------------------------------------
# morphology / edges
# --------------------------------------------------------------------------
# Oct 2026: large structuring elements.  scipy's binary_erosion/dilation precompute an
# offset table whose size grows with (element side)^4 on a 2-D image: a 150-px disk on a
# 400 x 400 slice needs ~3 GB and 8 s, a 200-px disk ~10 GB, an image-sized one ~27 GB.
# The visceral pass builds disks from the fitted ellipse (vabavg), so a runaway fit asks
# for the largest.  _row_morph gives the SAME result (same centre, reflection and border
# conventions as scipy; checked pixel for pixel on random images and elements of both
# parities, see the self-test in run_fatvol.py) in O(side x pixels) time and O(pixels)
# memory, by treating each row of the element as runs and taking windowed sums along the
# other axis.  Used for 2-D elements at least FAST_MORPH_MIN wide; set FAST_MORPH_MIN to
# None to always use scipy.
FAST_MORPH_MIN = 16


def _row_runs(struct):
    runs = []
    for i in range(struct.shape[0]):
        idx = np.flatnonzero(struct[i])
        if idx.size:
            for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                runs.append((i, int(seg[0]), int(seg[-1])))
    return runs


def _row_morph(img, struct, erosion):
    """Binary erosion (border 1) or dilation (border 0) of a 2-D image by a 2-D element,
    identical to scipy.ndimage.binary_erosion(..., border_value=1) / binary_dilation(...,
    border_value=0) with origin 0: the element's centre is index size//2 on each axis;
    erosion out[p] = AND_t in[p + t - c], dilation out[p] = OR_t in[p - (t - c)]."""
    a = np.asarray(img, bool)
    nx, ny = a.shape
    L0, L1 = struct.shape
    c0, c1 = L0 // 2, L1 // 2
    bv = 1 if erosion else 0
    P0, P1 = L0, L1
    pad = np.pad(a, ((P0, P0), (P1, P1)), constant_values=bv)
    cs = np.zeros((pad.shape[0], pad.shape[1] + 1), np.int32)
    np.cumsum(pad, axis=1, dtype=np.int32, out=cs[:, 1:])
    out = np.ones((nx, ny), bool) if erosion else np.zeros((nx, ny), bool)
    for i, j0, j1 in _row_runs(np.asarray(struct, bool)):
        di = i - c0
        if erosion:
            r0, lo = P0 + di, P1 + (j0 - c1)
            hi = P1 + (j1 - c1) + 1
            out &= (cs[r0:r0 + nx, hi:hi + ny] - cs[r0:r0 + nx, lo:lo + ny]) == (j1 - j0 + 1)
        else:
            r0, lo = P0 - di, P1 - (j1 - c1)
            hi = P1 - (j0 - c1) + 1
            out |= (cs[r0:r0 + nx, hi:hi + ny] - cs[r0:r0 + nx, lo:lo + ny]) > 0
    return out


def _use_rows(img, struct):
    return (FAST_MORPH_MIN is not None and np.ndim(img) == 2 and np.ndim(struct) == 2
            and max(np.shape(struct)) >= FAST_MORPH_MIN)


def erode(img: np.ndarray, struct: np.ndarray) -> np.ndarray:
    if _use_rows(img, struct):
        return _row_morph(img, struct, True).astype(np.uint8)
    return ndimage.binary_erosion(img.astype(bool), structure=struct.astype(bool),
                                  border_value=1).astype(np.uint8)


def dilate(img: np.ndarray, struct: np.ndarray) -> np.ndarray:
    if _use_rows(img, struct):
        return _row_morph(img, struct, False).astype(np.uint8)
    return ndimage.binary_dilation(img.astype(bool), structure=struct.astype(bool),
                                   border_value=0).astype(np.uint8)


def opening(img: np.ndarray, struct: np.ndarray) -> np.ndarray:
    """IDL `dilate(erode(x, s), s)`."""
    return dilate(erode(img, struct), struct)


def closing(img: np.ndarray, struct: np.ndarray) -> np.ndarray:
    """IDL `erode(dilate(x, s), s)`."""
    return erode(dilate(img, struct), struct)


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """IDL SOBEL(x) — gradient magnitude, unthresholded."""
    a = img.astype(np.float64)
    gx = ndimage.sobel(a, axis=0, mode="nearest")
    gy = ndimage.sobel(a, axis=1, mode="nearest")
    return np.hypot(gx, gy)


def sobel_edge(img: np.ndarray) -> np.ndarray:
    """IDL `sobel(x*100.) gt 0` — a binary edge map."""
    return (sobel_magnitude(img.astype(np.float64) * 100.0) > 0).astype(np.uint8)


def morph_distance(mask: np.ndarray) -> np.ndarray:
    """IDL MORPH_DISTANCE(..., NEIGHBOR_SAMPLING=3) — Euclidean distance transform."""
    return ndimage.distance_transform_edt(mask.astype(bool))


def conditional_dilate(seed: np.ndarray, mask: np.ndarray,
                       struct: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Geodesic dilation: grow `seed` inside `mask` until stable.

    Reproduces the IDL `while done eq 0 do tmp = dilate(tmp, s) and mask` loops.
    `max_iter` is a guard the original lacked; it cannot change a converged result.
    """
    seed = np.asarray(seed)
    mask = np.asarray(mask)
    st = struct.astype(bool)

    # The .pro applies a 2D structuring element to a 3D volume in one place
    # (the peripheral-fat island cleanup). IDL's DILATE nominally requires
    # matching ranks, so that call was relying on undefined behaviour. Slices
    # are treated independently here, which is the only reading that makes
    # anatomical sense: fat islands are separated within a slice, not across
    # a 10 mm slice gap.
    if seed.ndim == 3 and st.ndim == 2:
        out = np.zeros_like(seed, dtype=np.uint8)
        for k in range(seed.shape[2]):
            out[:, :, k] = conditional_dilate(seed[:, :, k], mask[:, :, k],
                                              struct, max_iter)
        return out

    cur = (seed.astype(bool) & mask.astype(bool))
    for _ in range(max_iter):
        nxt = ndimage.binary_dilation(cur, structure=st) & mask.astype(bool)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur.astype(np.uint8)


# --------------------------------------------------------------------------
# thresholding
# --------------------------------------------------------------------------
def _idl_histogram(data: np.ndarray, vmin: float, nbins: int):
    """IDL HISTOGRAM(data, MIN=vmin, NBINS=nbins) with implicit binsize.

    IDL derives binsize = (max - min) / (nbins - 1) when BINSIZE is absent.
    Returns (counts, binsize).
    """
    vmax = float(np.max(data))
    binsize = (vmax - vmin) / (nbins - 1) if nbins > 1 else 1.0
    if binsize <= 0:
        return np.zeros(nbins, dtype=np.int64), 1.0
    idx = np.floor((data.ravel() - vmin) / binsize).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < nbins)]
    return np.bincount(idx, minlength=nbins), binsize


def _thresh_valley(h: np.ndarray) -> float:
    """Method 0 — lowest point between the two histogram peaks."""
    n = len(h)
    half = n // 2
    imaxh = int(np.argmax(h))
    if imaxh >= half:
        imaxh2 = int(np.argmax(h[:half]))
        return float(np.argmin(h[imaxh2:imaxh + 1]) + imaxh2)
    imaxh2 = int(np.argmax(h[half:])) + half
    return float(np.argmin(h[imaxh:imaxh2 + 1]) + imaxh)


def _thresh_isodata(h: np.ndarray, idl_integer_division: bool = True) -> float:
    """Method 1 — iterative intermeans (ISODATA), the original default.

    NOTE ON FIDELITY: in the .pro, tempSum1..4 are declared `long(0)`, so
    `tempSum1/tempSum2` is INTEGER division in IDL; only the following `/2.0`
    promotes to float. That truncation shifts the threshold slightly. It is
    reproduced when `idl_integer_division=True` (default) so results match the
    original. Set False for the mathematically clean version.
    """
    nbins = len(h)
    i = np.arange(nbins, dtype=np.int64)
    moving = 0
    thresh = float(nbins)
    while moving + 1 <= thresh and moving <= nbins - 1:
        s1 = int(np.sum(i[:moving + 1] * h[:moving + 1]))
        s2 = int(np.sum(h[:moving + 1]))
        s3 = int(np.sum(i[moving + 1:] * h[moving + 1:]))
        s4 = int(np.sum(h[moving + 1:]))
        # IDL integer division by zero yields 0 (with a warning), and the
        # original relied on that: for every bin below the first occupied one
        # tempSum2 is 0, so the lower term contributes nothing and the loop
        # keeps going. Breaking out here instead would return thresh = nbins
        # and produce an empty fat mask.
        lo = 0.0 if s2 == 0 else ((s1 // s2) if idl_integer_division else (s1 / s2))
        hi = 0.0 if s4 == 0 else ((s3 // s4) if idl_integer_division else (s3 / s4))
        thresh = lo / 2.0 + hi / 2.0
        moving += 1
    return thresh


def gray_threshold(data: np.ndarray, method: int = 1, nbins: int = 100,
                   idl_integer_division: bool = True) -> float:
    """Return a threshold in GRAY LEVELS, as the IDL `thregray` variable."""
    h, binsize = _idl_histogram(data, vmin=1, nbins=nbins)
    if h.sum() == 0:
        return 0.0
    if method == 0:
        t = _thresh_valley(h)
    elif method == 1:
        t = _thresh_isodata(h, idl_integer_division)
    else:
        raise ValueError("The thresholding method is not supported.")
    return float(round(binsize * t))


# --------------------------------------------------------------------------
# coordinate masks
# --------------------------------------------------------------------------
def coordinate_masks(nx: int, ny: int, pxsize: float, pysize: float):
    """radmask (mm), angmask (deg, -180..180), xmask, ymask, imask, jmask.

    Angles are measured about the IMAGE centre, not the body centre — the
    original comments this repeatedly and downstream logic depends on it.
    """
    xcenter = nx // 2
    ycenter = ny // 2
    i = np.arange(nx)[:, None]
    j = np.arange(ny)[None, :]
    xmask = (i - xcenter) * pxsize * np.ones((1, ny))
    ymask = (ycenter - 1 - j) * pysize * np.ones((nx, 1))
    radmask = np.hypot(xmask, ymask)
    angmask = np.degrees(np.arctan2(ymask, xmask))
    imask = np.broadcast_to(i, (nx, ny)).copy()
    jmask = np.broadcast_to(j, (nx, ny)).copy()
    return radmask, angmask, xmask, ymask, imask, jmask


def polar_outer_edge(edge: np.ndarray, radmask: np.ndarray, angmask: np.ndarray):
    """For each integer angle -179..180, the outermost edge pixel.

    Returns (r[360], flat_index[360], present[360]). Vectorised replacement for
    the IDL `for i = -179, 180 do idx = where(edgeth eq i)` loops.
    """
    r = np.zeros(360)
    idxs = np.zeros(360, dtype=np.int64)
    present = np.zeros(360, dtype=bool)
    sel = edge.astype(bool)
    if not sel.any():
        return r, idxs, present
    flat = np.flatnonzero(sel)
    th = np.round(angmask.ravel()[flat]).astype(int)
    rr = radmask.ravel()[flat]
    bin_ = th + 179
    ok = (bin_ >= 0) & (bin_ < 360)
    flat, rr, bin_ = flat[ok], rr[ok], bin_[ok]
    order = np.lexsort((rr, bin_))          # ascending r within each angle bin
    bin_s, rr_s, flat_s = bin_[order], rr[order], flat[order]
    last = np.ones(len(bin_s), dtype=bool)
    last[:-1] = bin_s[:-1] != bin_s[1:]     # keep the largest r per bin
    r[bin_s[last]] = rr_s[last]
    idxs[bin_s[last]] = flat_s[last]
    present[bin_s[last]] = True
    return r, idxs, present


def mask_from_radial_profile(profile: np.ndarray, radmask: np.ndarray,
                             angmask: np.ndarray) -> np.ndarray:
    """Fill every pixel whose radius is within profile(theta).

    Replaces the triple-nested per-pixel INTERPOL loop near the end of the .pro,
    which was the slowest thing in the program by a wide margin.
    """
    thi = np.arange(360) - 179.0
    rr = np.interp(angmask.ravel(), thi, profile, left=profile[0], right=profile[-1])
    return (radmask.ravel() <= rr).reshape(radmask.shape).astype(np.uint8)
