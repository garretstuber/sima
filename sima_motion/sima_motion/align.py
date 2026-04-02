"""Cross-correlation alignment utilities.

Adapted from CellProfiler (GNU GPL).
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
"""

import warnings

import numpy as np

try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
    from pyfftw.interfaces.numpy_fft import rfftn, irfftn
except ImportError:
    from scipy.fftpack import fft2, ifft2
    from numpy.fft import rfftn, irfftn

try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

import scipy.ndimage as scind
import scipy.sparse


def cross_correlation_3d(pixels1, pixels2):
    """Align two 3D images using max normalized cross-correlation.

    Returns the z,y,x offsets to add to image1's indexes to align it with
    image2.

    Based on "Fast Normalized Cross-Correlation" by J.P. Lewis.
    """
    s = np.maximum(pixels1.shape, pixels2.shape)
    fshape = s * 2

    i, j, k = np.mgrid[-s[0]:s[0], -s[1]:s[1], -s[2]:s[2]]
    unit = np.abs(i * j * k).astype(float)
    unit[unit < 1] = 1

    pixels1 = np.nan_to_num(pixels1 - nanmean(pixels1))
    pixels2 = np.nan_to_num(pixels2 - nanmean(pixels2))

    fp1 = rfftn(pixels1.astype('float32'), fshape, axes=(0, 1, 2))
    fp2 = rfftn(pixels2.astype('float32'), fshape, axes=(0, 1, 2))
    corr12 = irfftn(fp1 * fp2.conj(), axes=(0, 1, 2)).real

    def get_cumsums(im, fshape):
        im_si = im.shape[0]
        im_sj = im.shape[1]
        im_sk = im.shape[2]
        im_sum = np.zeros(fshape)
        im_sum[:im_si, :im_sj, :im_sk] = cumsum_quadrant(
            im, False, False, False)
        im_sum[:im_si, :im_sj, -im_sk:] = cumsum_quadrant(
            im, False, False, True)
        im_sum[:im_si, -im_sj:, :im_sk] = cumsum_quadrant(
            im, False, True, True)
        im_sum[:im_si, -im_sj:, -im_sk:] = cumsum_quadrant(
            im, False, True, False)
        im_sum[-im_si:, :im_sj, :im_sk] = cumsum_quadrant(
            im, True, False, True)
        im_sum[-im_si:, :im_sj, -im_sk:] = cumsum_quadrant(
            im, True, False, False)
        im_sum[-im_si:, -im_sj:, :im_sk] = cumsum_quadrant(
            im, True, True, True)
        im_sum[-im_si:, -im_sj:, -im_sk:] = cumsum_quadrant(
            im, True, True, False)
        return im_sum / unit

    p1_mean = get_cumsums(pixels1, fshape)
    p2_mean = get_cumsums(pixels2, fshape)

    p1sd = np.sum(pixels1 ** 2) - p1_mean ** 2 * np.prod(s)
    p2sd = np.sum(pixels2 ** 2) - p2_mean ** 2 * np.prod(s)

    sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrnorm = corr12 / sd

    corrnorm[(unit < np.prod(s) / 2) &
             (sd < np.mean(sd) / 100)] = 0
    corrnorm[unit < np.prod(s) / 4] = 0

    return corrnorm


def cross_correlation_2d(pixels1, pixels2):
    """Align two 2D images using max normalized cross-correlation.

    Based on "Fast Normalized Cross-Correlation" by J.P. Lewis.
    """
    s = np.maximum(pixels1.shape, pixels2.shape)
    fshape = s * 2

    i, j = np.mgrid[-s[0]:s[0], -s[1]:s[1]]
    unit = np.abs(i * j).astype(float)
    unit[unit < 1] = 1

    pixels1 = np.nan_to_num(pixels1 - nanmean(pixels1))
    pixels2 = np.nan_to_num(pixels2 - nanmean(pixels2))

    fp1 = fft2(pixels1.astype('float32'), fshape)
    fp2 = fft2(pixels2.astype('float32'), fshape)
    corr12 = ifft2(fp1 * fp2.conj()).real

    p1_si = pixels1.shape[0]
    p1_sj = pixels1.shape[1]
    p1_sum = np.zeros(fshape)
    p1_sum[:p1_si, :p1_sj] = cumsum_quadrant(pixels1, False, False)
    p1_sum[:p1_si, -p1_sj:] = cumsum_quadrant(pixels1, False, True)
    p1_sum[-p1_si:, :p1_sj] = cumsum_quadrant(pixels1, True, False)
    p1_sum[-p1_si:, -p1_sj:] = cumsum_quadrant(pixels1, True, True)
    p1_mean = p1_sum / unit

    p2_si = pixels2.shape[0]
    p2_sj = pixels2.shape[1]
    p2_sum = np.zeros(fshape)
    p2_sum[:p2_si, :p2_sj] = cumsum_quadrant(pixels2, False, False)
    p2_sum[:p2_si, -p2_sj:] = cumsum_quadrant(pixels2, False, True)
    p2_sum[-p2_si:, :p2_sj] = cumsum_quadrant(pixels2, True, False)
    p2_sum[-p2_si:, -p2_sj:] = cumsum_quadrant(pixels2, True, True)
    p2_sum = np.fliplr(np.flipud(p2_sum))
    p2_mean = p2_sum / unit

    p1sd = np.sum(pixels1 ** 2) - p1_mean ** 2 * np.prod(s)
    p2sd = np.sum(pixels2 ** 2) - p2_mean ** 2 * np.prod(s)

    sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrnorm = corr12 / sd

    corrnorm[(unit < np.prod(s) / 2) &
             (sd < np.mean(sd) / 100)] = 0
    corrnorm[unit < np.prod(s) / 4] = 0
    return corrnorm


def align_cross_correlation(pixels1, pixels2, displacement_bounds=None):
    """Align the second image with the first using max cross-correlation.

    Returns the y,x offsets to add to image1's indexes to align it with
    image2, as well as the correlation at that offset.
    """
    s = np.maximum(pixels1.shape[:-1], pixels2.shape[:-1])
    fshape = s * 2
    if len(s) == 2:
        corr = cross_correlation_2d
    elif len(s) == 3:
        corr = cross_correlation_3d
    else:
        raise ValueError

    corrnorm = sum(corr(pixels1[..., c], pixels2[..., c])
                   for c in range(pixels1.shape[-1])) / pixels1.shape[-1]
    offset = fshape - np.array(pixels1.shape[:-1])
    for i in range(corrnorm.ndim):
        corrnorm = np.roll(corrnorm, offset[i], axis=i)

    if displacement_bounds is not None:
        idx_bounds = displacement_bounds + offset
        corrnorm[:idx_bounds[0][0]] = -np.inf
        corrnorm[idx_bounds[1][0]:] = -np.inf
        corrnorm[:, :idx_bounds[0][1]] = -np.inf
        corrnorm[:, idx_bounds[1][1]:] = -np.inf
        if idx_bounds.shape[1] == 3:
            corrnorm[:, :, :idx_bounds[0][2]] = -np.inf
            corrnorm[:, :, idx_bounds[1][2]:] = -np.inf

    idx = np.unravel_index(np.argmax(corrnorm), fshape)
    return np.array(idx) - offset, corrnorm[idx]


def cumsum_quadrant(x, i_forwards, j_forwards, k_forwards=None):
    """Return the cumulative sum going in the i, then j direction."""
    if i_forwards:
        x = x.cumsum(0)
    else:
        x = np.flipud(np.flipud(x).cumsum(0))
    if j_forwards:
        x = x.cumsum(1)
    else:
        x = np.fliplr(np.fliplr(x).cumsum(1))
    if k_forwards is None:
        return x
    if k_forwards:
        return x.cumsum(2)
    else:
        return x[:, :, ::-1].cumsum(2)[:, :, ::-1]


def entropy(x):
    """The entropy of x as if x is a probability distribution."""
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram != 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def entropy2(x, y):
    """Joint entropy of paired samples X and Y."""
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    xy = 256 * x + y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix((np.ones(xy.shape),
                                      (xy, np.zeros(xy.shape))))
    histogram = sparse.toarray()
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram > 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def reshape_image(source, new_shape):
    """Reshape an image to a larger shape, padding with zeros."""
    if tuple(source.shape) == tuple(new_shape):
        return source
    result = np.zeros(new_shape, source.dtype)
    result[:source.shape[0], :source.shape[1]] = source
    return result


def stretch(image, mask=None):
    """Normalize an image to make the minimum zero and maximum one."""
    image = np.array(image, float)
    if np.prod(image.shape) == 0:
        return image
    if mask is None:
        minval = np.min(image)
        maxval = np.max(image)
        if minval == maxval:
            if minval < 0:
                return np.zeros_like(image)
            elif minval > 1:
                return np.ones_like(image)
            return image
        else:
            return (image - minval) / (maxval - minval)
    else:
        significant_pixels = image[mask]
        if significant_pixels.size == 0:
            return image
        minval = np.min(significant_pixels)
        maxval = np.max(significant_pixels)
        if minval == maxval:
            transformed_image = minval
        else:
            transformed_image = (significant_pixels - minval) / (maxval - minval)
        image[mask] = transformed_image
        return image


def offset_slice(pixels1, pixels2, i, j):
    """Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.
    """
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return (p1, p2)
