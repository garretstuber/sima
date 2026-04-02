"""Discrete Fourier Transform registration for motion correction.

Efficient subpixel image registration by cross correlation. A reference image
is iteratively computed by aligning and averaging a subset of images/frames.

2015 Lloyd Russell, Christoph Schmidt-Hieber

Credit to Marius Pachitariu for concept of registering to aligned mean image.
Credit to Olivier Dupont-Therrien for Gaussian blur & Laplacian preprocessing.

Based on skimage.feature.register_translation, which is a port of MATLAB code
by Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
"Efficient subpixel image registration algorithms," Optics Letters 33,
156-158 (2008).
"""

from functools import partial
import multiprocessing
import time

import numpy as np
from scipy.ndimage import shift, laplace, gaussian_filter

from . import motion

try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
except ImportError:
    from numpy.fft import fftn, ifftn


class DiscreteFourier2D(motion.MotionEstimationStrategy):
    """Motion correction via efficient subpixel image registration.

    Parameters
    ----------
    upsample_factor : int, optional
        Upsampling factor. Default: 1.
    max_displacement : array of int, optional
        Maximum allowed displacement magnitudes in [y,x]. Default: None.
    num_images_for_mean : int, optional
        Number of images to use for aligned mean. Default: 100.
    randomise_frames : bool, optional
        Randomise images for mean. Default: True.
    err_thresh : float, optional
        Threshold for mean pixel offset convergence. Default: 0.01.
    max_iterations : int, optional
        Maximum iterations for aligned mean. Default: 5.
    n_processes : int, optional
        Number of workers. Default: 1.
    verbose : bool, optional
        Enable verbose mode. Default: False.
    return_registered : bool, optional
        Return registered frames. Default: False.
    laplace : float, optional
        Sigma for Gaussian blur & Laplacian preprocessing. Default: 0.0.
    """

    def __init__(self, upsample_factor=1, max_displacement=None,
                 num_images_for_mean=100,
                 randomise_frames=True, err_thresh=0.01, max_iterations=5,
                 rotation_scaling=False, save_fmt='mptiff', save_name=None,
                 n_processes=1, verbose=False, return_registered=False,
                 laplace=0.0):
        self._params = dict(locals())
        del self._params['self']

    def _estimate(self, dataset):
        params = self._params
        verbose = params['verbose']
        n_processes = params['n_processes']

        if verbose:
            print('Using ' + str(n_processes) + ' worker(s)')

        displacements = []

        for sequence in dataset:
            num_planes = sequence.shape[1]
            num_channels = sequence.shape[4]
            if num_channels > 1:
                raise NotImplementedError("Only one channel can be used "
                                          "for DFT motion correction.")

            for plane_idx in range(num_planes):
                if verbose:
                    print('Loading plane ' + str(plane_idx + 1) + ' of ' +
                          str(num_planes) + ' into numpy array')
                t0 = time.time()
                frames = np.array(sequence[:, plane_idx, :, :, 0])
                frames = np.squeeze(frames)
                e1 = time.time() - t0
                if verbose:
                    print('    Loaded in: ' + str(e1) + ' s')

                if params['laplace'] > 0:
                    framesl = np.array([
                        np.abs(laplace(gaussian_filter(frame, params['laplace'])))
                        for frame in frames])
                else:
                    framesl = frames
                output = _register(
                    framesl,
                    upsample_factor=params['upsample_factor'],
                    max_displacement=params['max_displacement'],
                    num_images_for_mean=params['num_images_for_mean'],
                    randomise_frames=params['randomise_frames'],
                    err_thresh=params['err_thresh'],
                    max_iterations=params['max_iterations'],
                    n_processes=params['n_processes'],
                    save_fmt=params['save_fmt'],
                    save_name=params['save_name'],
                    verbose=params['verbose'],
                    return_registered=params['return_registered'])

                if params['return_registered']:
                    dy, dx, registered_frames = output
                else:
                    dy, dx = output

                frame_shifts = np.zeros([len(frames), num_planes, 2])
                for idx, frame in enumerate(sequence):
                    frame_shifts[idx, plane_idx] = [dy[idx], dx[idx]]
            displacements.append(frame_shifts)

            total_time = time.time() - t0
            if verbose:
                print('    Total time for plane ' + str(plane_idx + 1) + ': ' +
                      str(total_time) + ' s')

        return displacements


def _register(frames, upsample_factor=1, max_displacement=None,
              num_images_for_mean=100, randomise_frames=True, err_thresh=0.01,
              max_iterations=5, rotation_scaling=False, save_fmt='mptiff',
              save_name=None, n_processes=1, verbose=False,
              return_registered=False):
    """Make aligned mean image and register each frame to it."""
    t0 = time.time()

    mean_img = _make_mean_img(frames,
                              num_images_for_mean=num_images_for_mean,
                              randomise_frames=randomise_frames,
                              err_thresh=err_thresh,
                              max_iterations=max_iterations,
                              upsample_factor=upsample_factor,
                              n_processes=n_processes,
                              max_displacement=max_displacement,
                              verbose=verbose)
    e1 = time.time() - t0
    if verbose:
        print('        Time taken: ' + str(e1) + ' s')

    output = _register_all_frames(frames, mean_img,
                                  upsample_factor=upsample_factor,
                                  n_processes=n_processes,
                                  max_displacement=max_displacement,
                                  verbose=verbose,
                                  return_registered=return_registered)

    if return_registered:
        dy, dx, registered_frames = output
    else:
        dy, dx = output

    e2 = time.time() - t0 - e1
    if verbose:
        print('        Time taken: ' + str(e2) + ' s')

    if return_registered:
        if save_name is not None and save_name != 'none':
            _save_registered_frames(registered_frames, save_name, save_fmt,
                                    verbose=verbose)

    total_time = time.time() - t0
    if verbose:
        print('    Completed in: ' + str(total_time) + ' s')

    if return_registered:
        return dy, dx, registered_frames
    else:
        return dy, dx


def _make_mean_img(frames, num_images_for_mean=100, randomise_frames=True,
                   err_thresh=0.01, max_iterations=5, upsample_factor=1,
                   n_processes=1, max_displacement=None, verbose=False):
    """Make an aligned mean image as the registration reference."""
    input_shape = frames.shape
    input_dtype = np.array(frames[0]).dtype

    if num_images_for_mean > input_shape[0]:
        num_images_for_mean = input_shape[0]

    frames_for_mean = np.zeros([num_images_for_mean, input_shape[1],
                               input_shape[2]], dtype=input_dtype)

    if randomise_frames:
        if verbose:
            print('    Making aligned mean image from ' +
                  str(num_images_for_mean) + ' random frames...')
        for idx, frame_num in enumerate(np.random.choice(input_shape[0],
                                        size=num_images_for_mean,
                                        replace=False)):
            frames_for_mean[idx] = frames[frame_num]
    else:
        if verbose:
            print('    Making aligned mean image from first ' +
                  str(num_images_for_mean) + ' frames...')
        frames_for_mean = frames[0:num_images_for_mean]

    mean_img = np.mean(frames_for_mean, 0)
    iteration = 1
    mean_img_err = 9999

    while mean_img_err > err_thresh and iteration < max_iterations:
        map_function = partial(_register_frame, mean_img=mean_img,
                               upsample_factor=upsample_factor,
                               max_displacement=max_displacement,
                               return_registered=True)

        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            results = pool.map(map_function, frames_for_mean)
            pool.close()
        else:
            results = map(map_function, frames_for_mean)

        mean_img_dx = np.zeros(num_images_for_mean, dtype=np.float64)
        mean_img_dy = np.zeros(num_images_for_mean, dtype=np.float64)

        for idx, result in enumerate(results):
            mean_img_dy[idx] = result[0]
            mean_img_dx[idx] = result[1]
            frames_for_mean[idx] = result[2]

        mean_img = np.mean(frames_for_mean, 0)
        mean_img_err = np.mean(
            np.absolute(mean_img_dx)) + np.mean(np.absolute(mean_img_dy))

        if verbose:
            print('        Iteration ' + str(iteration) +
                  ', average error: ' + str(mean_img_err) + ' pixels')
        iteration += 1

    return mean_img


def _register_all_frames(frames, mean_img, upsample_factor=1,
                         n_processes=1, max_displacement=None,
                         return_registered=False, verbose=False):
    """Register all input frames to the aligned mean image."""
    input_shape = frames.shape
    input_dtype = np.array(frames[0]).dtype

    if verbose:
        print('    Registering all ' + str(frames.shape[0]) + ' frames...')

    map_function = partial(_register_frame, mean_img=mean_img,
                           upsample_factor=upsample_factor,
                           max_displacement=max_displacement,
                           return_registered=return_registered)

    if n_processes > 1:
        pool = multiprocessing.Pool(n_processes)
        results = pool.map(map_function, frames)
        pool.close()
    else:
        results = map(map_function, frames)

    dx = np.zeros(input_shape[0], dtype=np.float64)
    dy = np.zeros(input_shape[0], dtype=np.float64)

    if return_registered:
        registered_frames = np.zeros([input_shape[0], input_shape[1],
                                     input_shape[2]], dtype=input_dtype)
        for idx, result in enumerate(results):
            dy[idx] = result[0]
            dx[idx] = result[1]
            registered_frames[idx] = result[2]
        return dy, dx, registered_frames
    else:
        for idx, result in enumerate(results):
            dy[idx] = result[0]
            dx[idx] = result[1]
        return dy, dx


def _register_frame(frame, mean_img, upsample_factor=1,
                    max_displacement=None, return_registered=False):
    """Register a single frame to the mean image."""
    dy, dx = _register_translation(mean_img, frame,
                                   upsample_factor=upsample_factor)

    if max_displacement is not None:
        if dy > max_displacement[0]:
            dy = max_displacement[0]
        if dx > max_displacement[1]:
            dx = max_displacement[1]

    if return_registered:
        registered_frame = shift(frame, [dy, dx], order=3, mode='constant',
                                 cval=0, output=frame.dtype)
        return dy, dx, registered_frame
    else:
        return dy, dx


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """Upsampled DFT by matrix multiplication.

    Based on skimage.feature.register_translation.
    """
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (np.fft.ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(data.shape[1] / 2)).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            np.fft.ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(data.shape[0] / 2))
    )

    row_kernel_dot = row_kernel.dot(data)
    return row_kernel_dot.dot(col_kernel)


def _compute_phasediff(cross_correlation_max):
    """Compute global phase difference between images."""
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """Compute RMS error metric between src_image and target_image."""
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() / \
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))


def _register_translation(src_image, target_image, upsample_factor=1,
                          space="real"):
    """Efficient subpixel image translation registration by cross-correlation.

    Based on skimage.feature.register_translation.
    """
    if src_image.shape != target_image.shape:
        raise ValueError("Images must be the same size for "
                         "register_translation")

    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("register_translation only supports "
                                  "subpixel registration for 2D images")

    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    elif space.lower() == 'real':
        src_image = np.array(src_image, dtype=np.complex128, copy=False)
        target_image = np.array(target_image, dtype=np.complex128, copy=False)
        src_freq = fftn(src_image)
        target_freq = fftn(target_image)
    else:
        raise ValueError("register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = ifftn(image_product)

    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
    else:
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()

        cross_correlation /= normalization
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts


def _save_registered_frames(frames, save_name, save_fmt, verbose=False):
    """Save registered frames to disk."""
    if verbose:
        print('    Saving...')
    try:
        import tifffile
    except ImportError:
        if verbose:
            print('        Cannot find tifffile')
        return

    if save_fmt == 'singles':
        for idx in range(frames.shape[0]):
            tifffile.imsave(
                save_name + '_' + '{number:05d}'.format(number=idx) +
                '_DFTreg.tif', frames[idx].astype(np.uint16))
    if save_fmt == 'mptiff':
        tifffile.imsave(save_name + '_DFTreg.tif',
                        frames.astype(np.uint16))
    elif save_fmt == 'bigtiff':
        tifffile.imsave(save_name + '_DFTreg.tif',
                        frames.astype(np.uint16), bigtiff=True)
