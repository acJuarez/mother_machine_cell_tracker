"""
Fluorescence denoising filters shared by the kymograph viewer and the
track-displacement analysis, so both apply identical filtering.

The main entry point is `denoise_fluor(stack, method, ...)`, which filters a
(T, Y, X) fluorescence stack frame-by-frame and preserves dtype.
"""

import numpy as np


def denoise_fluor(stack, method="median", size=3, sigma=1.0, nlm_h=1.15, otsu_scale=1.0):
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

    Returns a new array of the same shape/dtype (no in-place modification).
    """
    if stack is None or method == "none":
        return stack

    from scipy import ndimage

    if method == "nlm":
        from skimage.restoration import denoise_nl_means

    if method == "otsu":
        return _otsu_background_subtract(stack, size, otsu_scale)

    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        if method == "median":
            out[t] = ndimage.median_filter(stack[t], size=size)
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
