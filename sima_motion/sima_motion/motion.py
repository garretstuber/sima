"""Base motion estimation strategy and resonant correction.

Core abstractions for motion correction of fluorescence imaging data.
"""

import abc
import itertools as it

import numpy as np

from . import _motion as mc
from .sequence import ImagingDataset, Sequence, resolve_channels


def add_with_offset(array1, array2, offset):
    """Add array2 into array1 at the given offset.

    >>> from sima_motion.motion import add_with_offset
    >>> import numpy as np
    >>> a1 = np.zeros((4, 4))
    >>> a2 = np.ones((1, 2))
    >>> add_with_offset(a1, a2, (1, 2))
    >>> np.array_equal(a1[1:2, 2:4], a2)
    True

    """
    slices = tuple(slice(o, o + e) for o, e in zip(offset, array2.shape))
    array1[slices] += array2


class MotionEstimationStrategy(metaclass=abc.ABCMeta):

    @classmethod
    def _make_nonnegative(cls, displacements):
        min_displacement = np.nanmin(
            [np.nanmin(s.reshape(-1, s.shape[-1]), 0) for s in displacements],
            0)
        new_displacements = [d - min_displacement for d in displacements]
        min_shifts = np.nanmin([np.nanmin(s.reshape(-1, s.shape[-1]), 0)
                                for s in new_displacements], 0)
        assert np.all(min_shifts == 0)
        return new_displacements

    @abc.abstractmethod
    def _estimate(self, dataset):
        return

    def estimate(self, dataset):
        """Estimate the displacements for a dataset.

        Parameters
        ----------
        dataset : ImagingDataset

        Returns
        -------
        displacements : list of ndarray of int
        """
        shifts = self._estimate(dataset)
        assert np.any(np.all(x is not np.ma.masked for x in shift)
                      for shift in it.chain.from_iterable(shifts))
        assert np.all(
            np.all(x is np.ma.masked for x in shift) or
            not np.any(x is np.ma.masked for x in shift)
            for shift in it.chain.from_iterable(shifts))
        shifts = self._make_nonnegative(shifts)
        assert np.any(np.all(x is not np.ma.masked for x in shift)
                      for shift in it.chain.from_iterable(shifts))
        assert np.all(
            np.all(x is np.ma.masked for x in shift) or
            not np.any(x is np.ma.masked for x in shift)
            for shift in it.chain.from_iterable(shifts))
        return shifts

    def correct(self, dataset, savedir=None, channel_names=None, info=None,
                correction_channels=None, trim_criterion=None):
        """Create a motion-corrected dataset.

        Parameters
        ----------
        dataset : ImagingDataset or list of Sequence
            Dataset or sequences to be motion corrected.
        savedir : str, optional
            The directory used to store the dataset.
        channel_names : list of str, optional
            Names for the channels.
        info : dict, optional
            Data for the order and timing of the data acquisition.
        correction_channels : list of int, optional
            Channels to use for motion correction. By default, all channels.
        trim_criterion : float, optional
            Required fraction of frames during which a location must
            be within the field of view.

        Returns
        -------
        dataset : ImagingDataset
            The motion-corrected dataset.
        """
        sequences = [s for s in dataset]
        if correction_channels:
            correction_channels = [
                resolve_channels(c, channel_names, len(sequences[0]))
                for c in correction_channels]
            mc_sequences = [s[:, :, :, :, correction_channels]
                            for s in sequences]
        else:
            mc_sequences = sequences
        displacements = self.estimate(ImagingDataset(mc_sequences, None))
        disp_dim = displacements[0].shape[-1]
        max_disp = np.ceil(
            np.max(list(it.chain.from_iterable(d.reshape(-1, disp_dim)
                                               for d in displacements)),
                   axis=0)).astype(np.int64)
        frame_shape = np.array(sequences[0].shape)[1: -1]  # (z, y, x)
        if len(max_disp) == 2:
            frame_shape[1:3] += max_disp
        else:
            frame_shape += max_disp
        corrected_sequences = [s.apply_displacements(d, frame_shape)
                               for s, d in zip(sequences, displacements)]
        planes, rows, columns = _trim_coords(
            trim_criterion, displacements, sequences[0].shape[1:4],
            frame_shape)
        corrected_sequences = [
            s[:, planes, rows, columns] for s in corrected_sequences]
        return ImagingDataset(
            corrected_sequences, savedir, channel_names=channel_names)


class ResonantCorrection(MotionEstimationStrategy):
    """Motion estimation strategy for resonant scanner data.

    Addresses the issue where even and odd rows are collected while
    scanning in opposite directions.

    Parameters
    ----------
    base_strategy : MotionEstimationStrategy
        The underlying motion estimation strategy.
    offset : int
        Horizontal displacement to be added to odd rows.
    """

    def __init__(self, base_strategy, offset=0):
        self._base_strategy = base_strategy
        self._offset = offset

    def _estimate(self, dataset):
        if not next(iter(dataset)).shape[2] % 2 == 0:
            raise ValueError(
                'Resonant motion correction requires an even number of rows')
        downsampled_dataset = ImagingDataset(
            [Sequence.join(
                *it.chain.from_iterable(
                    (seq[:, :, ::2, :, c], seq[:, :, 1::2, :, c])
                    for c in range(seq.shape[4])))
             for seq in dataset],
            None)
        downsampled_displacements = self._base_strategy.estimate(
            downsampled_dataset)
        displacements = []
        for d_disps in downsampled_displacements:
            disps = np.repeat(d_disps, 2, axis=2)
            disps[:, :, :, 0] *= 2
            disps[:, :, 1::2, -1] += self._offset
            displacements.append(disps)
        return displacements


def _trim_coords(trim_criterion, displacements, raw_shape, untrimmed_shape):
    """The coordinates used to trim the corrected imaging data."""
    epsilon = 1e-8
    assert len(raw_shape) == 3
    assert len(untrimmed_shape) == 3
    if trim_criterion is None:
        trim_criterion = 1.
    if trim_criterion == 0.:
        trim_criterion = epsilon
    if not isinstance(trim_criterion, (float, int)):
        raise TypeError('Invalid type for trim_criterion')
    obs_counts = sum(_observation_counts(raw_shape, d, untrimmed_shape)
                     for d in it.chain.from_iterable(displacements))
    num_frames = sum(len(x) for x in displacements)
    occupancy = obs_counts.astype(float) / num_frames

    plane_occupancy = occupancy.sum(axis=2).sum(axis=1) / (
        raw_shape[1] * raw_shape[2])
    good_planes = plane_occupancy + epsilon > trim_criterion
    plane_min = np.nonzero(good_planes)[0].min()
    plane_max = np.nonzero(good_planes)[0].max() + 1

    row_occupancy = occupancy.sum(axis=2).sum(axis=0) / (
        raw_shape[0] * raw_shape[2])
    good_rows = row_occupancy + epsilon > trim_criterion
    row_min = np.nonzero(good_rows)[0].min()
    row_max = np.nonzero(good_rows)[0].max() + 1

    col_occupancy = occupancy.sum(axis=1).sum(axis=0) / np.prod(
        raw_shape[:2])
    good_cols = col_occupancy + epsilon > trim_criterion
    col_min = np.nonzero(good_cols)[0].min()
    col_max = np.nonzero(good_cols)[0].max() + 1

    rows = slice(row_min, row_max)
    columns = slice(col_min, col_max)
    planes = slice(plane_min, plane_max)
    return planes, rows, columns


def _observation_counts(raw_shape, displacements, untrimmed_shape):
    cnt = np.zeros(untrimmed_shape, dtype=int)
    if displacements.ndim == 1:
        z, y, x = displacements
        cnt[z:(z + raw_shape[0]),
            y:(y + raw_shape[1]),
            x:(x + raw_shape[2])] = 1
    elif displacements.ndim == 2:
        for plane in range(raw_shape[0]):
            d = list(displacements[plane])
            if len(d) == 2:
                d = [0] + d
            d = np.round(np.array(d)).astype(int)
            cnt[plane + d[0],
                d[1]:(d[1] + int(np.round(raw_shape[1]))),
                d[2]:(d[2] + int(np.round(raw_shape[2])))] += 1
    elif displacements.ndim == 3:
        if displacements.shape[-1] == 2:
            return mc.observation_counts(raw_shape, displacements,
                                         untrimmed_shape)
        else:
            for plane, p_disp in enumerate(displacements):
                for row, r_disp in enumerate(p_disp):
                    add_with_offset(cnt, np.ones((1, 1, raw_shape[2])),
                                    r_disp + np.array([plane, row, 0]))
    else:
        raise ValueError
    return cnt
