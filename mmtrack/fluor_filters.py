"""
Fluorescence denoising filters shared by the kymograph viewer and the
track-displacement analysis. They call the same functions but not with the same
defaults: the viewer renders `fiji_median`, while the exported `fluor_filtered`
column is still `median`/3. The viewer layer and that column deliberately do not
match -- see CLAUDE.md.

The main entry point is `denoise_fluor(stack, method, ...)`, which filters a
(T, Y, X) fluorescence stack frame-by-frame and preserves dtype.

Filters only -- no display-range mapping lives here. The percentile black/white point
that the viewer's FIJI-equivalent layer applies is in
`generate_kymograph_viewer.fluor_display_range`, so the analysis path can never pick
up a display clip. The split costs nothing: clipping is monotone non-decreasing and an
odd-count median commutes with any such function, so clip-then-median and
median-then-clip give the same image.
"""

import numpy as np


def denoise_fluor(stack, method="median", size=3, sigma=1.0, nlm_h=1.15, otsu_scale=1.0,
                  radius=2.0):
    """Denoise a (T, Y, X) fluorescence stack frame-by-frame, preserving dtype.

    method:
        "none"     - return the stack unchanged.
        "median"   - median filter with square footprint `size` (removes speckle,
                     preserves cell edges).
        "mean"     - box average with square footprint `size` (linear; better for
                     Gaussian noise but blurs edges / smears hot pixels).
        "gaussian" - Gaussian blur with standard deviation `sigma`.
        "nlm"      - non-local means (edge-preserving, strongest noise reduction).
                     Strength scales with `nlm_h` (higher = smoother).
        "otsu"     - light median (footprint `size`) then background subtraction using a
                     single global Otsu threshold (scaled by `otsu_scale`); clips to 0.
                     Suppresses background haze rather than smoothing within the signal.
        "fiji_median" - median over ImageJ's circular footprint of `radius`
                     (Process > Filters > Median), edges replicated as ImageJ does.
                     radius=2.0 is a 21-point disk (x^2 + y^2 <= 5) -- NOT a 5x5 square
                     (25 points) and NOT skimage.morphology.disk(2) (13 points, which
                     is ImageJ radius 1.75). `size`, `sigma`, `nlm_h` and `otsu_scale`
                     are ignored.

    Returns a new array of the same shape/dtype (no in-place modification).
    """
    if stack is None or method == "none":
        return stack

    from scipy import ndimage

    if method == "nlm":
        from skimage.restoration import denoise_nl_means

    if method == "otsu":
        return _otsu_background_subtract(stack, size, otsu_scale)

    # Depends only on `radius`, so build it once instead of per frame.
    footprint = _imagej_disk(radius) if method == "fiji_median" else None

    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        if method == "median":
            out[t] = ndimage.median_filter(stack[t], size=size)
        elif method == "fiji_median":
            # Per frame, which is right for a stack. The kymograph viewer deliberately
            # does NOT use this path -- it filters the assembled kymograph instead, so
            # the kernel spans frame seams the way FIJI's does. See imagej_median.
            out[t] = imagej_median(stack[t], footprint=footprint)
        elif method == "mean":
            filtered = ndimage.uniform_filter(stack[t].astype(np.float32), size=size)
            out[t] = filtered.astype(stack.dtype)
        elif method == "gaussian":
            filtered = ndimage.gaussian_filter(stack[t].astype(np.float32), sigma=sigma)
            out[t] = filtered.astype(stack.dtype)
        elif method == "nlm":
            frame = stack[t].astype(np.float32)
            sig = _estimate_noise_sigma(frame)
            if sig <= 0:
                out[t] = stack[t]
                continue
            denoised = denoise_nl_means(
                frame, h=nlm_h * sig, sigma=sig,
                fast_mode=True, patch_size=5, patch_distance=6,
                preserve_range=True,
            )
            out[t] = denoised.astype(stack.dtype)
        else:
            raise ValueError(f"Unknown fluor filter method: {method!r}")
    return out


def to_kymograph(stack):
    """(T, Y, X) -> (Y, T*X): frames concatenated along x, nothing else.

    The single definition of the kymograph layout, shared by the viewer and the
    per-cell extractor so they cannot drift. Identical to what
    `mmtrack.plot_cells.create_kymograph` writes to `subtracted/fluor_kymos/*.tif`
    (`np.concatenate([stack[i] for i in range(T)], axis=1)`) -- verified bit-for-bit
    against that file -- which is why filtering the output of this reproduces a FIJI
    median run on those TIFFs. Frame t occupies columns [t*X, (t+1)*X).
    """
    t, y, x = stack.shape
    return stack.transpose(1, 0, 2).reshape(y, -1)


def from_kymograph(kymo, n_frames):
    """Inverse of `to_kymograph`: (Y, T*X) -> (T, Y, X).

    Needed because the FIJI transforms run on the assembled kymograph (the kernel must
    cross frame seams) while per-cell masking is per frame.
    """
    y, w = kymo.shape
    return kymo.reshape(y, n_frames, w // n_frames).transpose(1, 0, 2)


def remove_speckle(img, min_size, thr=None, connectivity=1):
    """Zero out connected components smaller than `min_size`, keeping grayscale elsewhere.

    Thresholds at `thr` to decide what an "object" is, finds components below
    `min_size`, and zeroes ONLY those pixels. Everything else -- including every pixel
    below the threshold -- is returned untouched, so this is a noise gate and not a
    binarisation of the dynamic range.

    `thr` decides what an object is; Otsu is used when it is None. The kymograph viewer
    passes the layer's DISPLAY MINIMUM (0.25 x max), so "object" means *visible* and the
    specks removed are ones you can actually see. Gating on Otsu instead makes the step
    inert under that black point -- measured 0.00% of rendered pixels change, because
    Otsu sits at 3 counts on a dim trench while nothing below 9 is displayed at all. A
    bare ">0" threshold fails for a different reason: these stacks are
    background-subtracted and clipped at 0, so it covers 78% of the image as ONE
    component of 528,491 px, of which only 0.43% is ever removable.

    connectivity=1 matches the project's labeling convention
    (mmtrack/cell_segmentation.py:100,104,144).

    `min_size` is an absolute pixel count, and at a display-minimum threshold that is
    safe: visible components are bright cell cores (median 22-53 px even on the ~1/3
    linear scale DUMM_giTG66_Glucose_012325), so a 10 px floor costs 1.1-4.1% of visible
    real-cell pixels there and 0.0% on bright trenches. An absolute floor at an OTSU
    threshold is a different story -- it cost 46.8% on that run, because Otsu's low cut
    produces many fragments whose size tracks cell area. That is why the scale-relative
    helper this once used is gone.

    Degenerate inputs pass through unchanged. Note skimage 0.25's threshold_otsu does NOT
    raise on a single-valued image -- it returns that value -- so the try/except is a
    guard for other failure modes, not for the constant case; what makes constant and
    all-zero images safe is that `img > thr` is then empty, so no component is small and
    nothing is zeroed.
    """
    import warnings

    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    if min_size is None or min_size < 1:
        return img
    if thr is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thr = threshold_otsu(img)
        except (ValueError, RuntimeError):
            return img
    mask = img > thr
    kept = remove_small_objects(mask, min_size=int(min_size),
                                connectivity=connectivity)
    # Zero ONLY the pixels of too-small components. Returning `where(kept, img, 0)`
    # instead would also zero everything BELOW the threshold, i.e. hard-threshold the
    # image and throw away the dim gradient around every cell -- measured at 61% of a
    # dim kymograph and 38% of a bright one, against 0.1-4% for the speckle alone. That
    # is a binarisation, not a noise gate.
    return np.where(mask & ~kept, 0, img)



def imagej_median(img, radius=2.0, footprint=None):
    """Median over ImageJ's circular footprint, on a 2-D image.

    The single implementation of the filter, so its two callers cannot drift:
    `denoise_fluor(method="fiji_median")` applies it frame-by-frame to a (T, Y, X)
    stack, and the kymograph viewer applies it to the ASSEMBLED 2-D kymograph.

    Those are not the same image. On an assembled kymograph, frames are concatenated
    along x, so at a frame boundary the kernel reaches into the adjacent TIME POINT --
    which is exactly what a FIJI median on `subtracted/fluor_kymos/<fov>_<peak>.tif`
    does, and is why the viewer wants it. Measured on 008/1488 (20-px frames, kernel
    +/-2): the two scopes differ on 8.3% of pixels, confined to columns 0, 1, 18 and 19
    of every frame, by up to 67 grey levels after the display map.

    mode="nearest" replicates the edge row/pixel the way ImageJ's RankFilters does
    (readLineToCacheOrPad); scipy's default "reflect" does not. Pass `footprint` to
    reuse one across frames instead of rebuilding it.
    """
    from scipy import ndimage

    if footprint is None:
        footprint = _imagej_disk(radius)
    return ndimage.median_filter(img, footprint=footprint, mode="nearest")


def _imagej_disk(radius):
    """ImageJ RankFilters' circular footprint for `radius`, as a bool array.

    Transcribed from ij/plugin/filter/RankFilters.makeLineRadii: r2 = int(radius^2) + 1,
    kradius = int(sqrt(r2)), and row y spans |x| <= int(sqrt(r2 - y^2)). That per-row
    construction is exactly `x^2 + y^2 <= r2` over the (2*kradius+1)^2 grid, so the
    integer form below is equivalent and needs no epsilon.

    radius 2.0 -> 21 points, row widths 3,5,5,5,3. Deliberately not
    skimage.morphology.disk(2), which is 13 points and is bit-identical to ImageJ
    radius 1.75 -- a visibly weaker smooth, not the filter FIJI applied.

    The two threshold promotions are ImageJ's own, kept so other radii stay faithful;
    neither fires at radius 2.0.
    """
    if 1.5 <= radius < 1.75:
        radius = 1.75
    elif 2.5 <= radius < 2.85:
        radius = 2.85
    r2 = int(radius * radius) + 1
    kradius = int(np.sqrt(r2))
    y, x = np.mgrid[-kradius:kradius + 1, -kradius:kradius + 1]
    return (x * x + y * y) <= r2


def _estimate_noise_sigma(frame):
    """Fast image-noise stddev estimate (Immerkaer 1996) — no PyWavelets needed.

    Convolves with a Laplacian kernel that suppresses smooth structure and leaves
    noise, then scales the mean absolute response. Robust enough to auto-set the
    non-local means strength per frame.
    """
    from scipy.ndimage import convolve

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = convolve(frame.astype(np.float32), kernel, mode="reflect")
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(conv)) / 6.0)


def _otsu_background_subtract(stack, size, otsu_scale):
    """Light median smooth, then subtract a single global Otsu threshold, clip to 0.

    A single threshold is computed over the whole (median-smoothed) stack so the
    background level is consistent across time (no per-frame flicker). Mirrors the
    scaled-Otsu pattern in mmtrack/cell_segmentation.py.
    """
    import warnings

    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smoothed = np.empty(stack.shape, dtype=np.float32)
    for t in range(stack.shape[0]):
        smoothed[t] = ndimage.median_filter(stack[t].astype(np.float32), size=size)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thr = otsu_scale * threshold_otsu(smoothed)
    except (ValueError, RuntimeError):
        # Constant/degenerate stack — nothing to threshold; return the smoothed data.
        return smoothed.astype(stack.dtype)

    subtracted = np.clip(smoothed - thr, 0, None)
    return subtracted.astype(stack.dtype)
