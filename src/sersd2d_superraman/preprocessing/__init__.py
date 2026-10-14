"""Explicit transforms for already preprocessed spectra."""

import numpy as np

CLASSIFIER_TRANSFORM = "per_spectrum_minmax_minus1_plus1_v1"


def classifier_input(spectra):
    x = np.asarray(spectra, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 1400 or not np.isfinite(x).all():
        raise ValueError("Classifier inputs must be finite [N,1400] spectra")
    low = x.min(axis=1, keepdims=True)
    span = x.max(axis=1, keepdims=True) - low
    if np.any(span <= 1e-8):
        raise ValueError("Cannot scale constant translated or original spectra")
    return np.ascontiguousarray((2 * (x - low) / span - 1)[:, None, :], dtype=np.float32)
