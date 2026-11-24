import numpy as np
from scipy import ndimage

def describe(volume):
    a = np.asarray(getattr(volume, 'data', volume))
    out = {}
    for label in np.unique(a):
        mask = a == label
        regions, n = ndimage.label(mask)
        counts = np.bincount(regions.ravel())[1:]
        spans = []
        for axis in range(3):
            spans.append(bool(set(np.unique(np.take(regions, 0, axis))) &
                              set(np.unique(np.take(regions, -1, axis))) - {0}))
        out[int(label)] = {'fraction': float(mask.mean()), 'components': int(n),
            'largest_component_fraction': float(counts.max() / mask.sum()),
            'spans_zyx': spans}
    return out

def two_point(volume, phase=1, axis=0):
    """Nonperiodic directional S2 with unbiased overlap normalization."""
    a = np.asarray(getattr(volume, 'data', volume)) == phase
    n = a.shape[axis]
    f = np.fft.rfft(a, n=2*n, axis=axis)
    corr = np.fft.irfft(f * f.conj(), n=2*n, axis=axis)
    corr = np.take(corr, np.arange(n), axis=axis)
    return corr.mean(axis=tuple(i for i in range(a.ndim) if i != axis)) / np.arange(n, 0, -1)

def compare(reference, generated):
    a = np.asarray(getattr(reference, 'data', reference))
    b = np.asarray(getattr(generated, 'data', generated))
    out = {}
    for phase in np.union1d(np.unique(a), np.unique(b)):
        errors = []
        for axis in range(3):
            x, y = two_point(a, phase, axis), two_point(b, phase, axis)
            n = min(len(x), len(y))
            errors.append(float(np.sqrt(np.mean((x[:n] - y[:n])**2))))
        out[int(phase)] = {'fraction_error': float(abs((a == phase).mean() - (b == phase).mean())),
                          's2_rmse_zyx': errors}
    return out
