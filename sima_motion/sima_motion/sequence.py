"""Minimal sequence and dataset abstractions for motion correction.

Extracted from SIMA (Kaifosh et al., 2014) with only the functionality
needed by the motion correction module.
"""

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np

from ._motion import _align_frame


def resolve_channels(chan, channel_names, num_channels=None):
    """Return the index corresponding to the channel.

    Parameters
    ----------
    chan : int or str
        Channel index or name.
    channel_names : list of str
        Names of available channels.
    num_channels : int, optional
        Total number of channels.

    Returns
    -------
    int or None
    """
    if chan is None:
        return None
    if num_channels is None:
        num_channels = len(channel_names)
    if isinstance(chan, int):
        if chan >= num_channels:
            raise ValueError('Invalid channel index.')
        return chan
    else:
        try:
            return channel_names.index(chan)
        except ValueError:
            raise ValueError('No channel exists with the specified name.')


class Sequence(metaclass=ABCMeta):
    """Object containing data from sequentially acquired imaging data.

    Sequences are array-like objects where each element is a frame with shape
    (num_planes, num_rows, num_columns, num_channels).

    Attributes
    ----------
    shape : tuple
        (num_frames, num_planes, num_rows, num_columns, num_channels)
    """

    def __getitem__(self, indices):
        """Create a new Sequence by slicing this Sequence."""
        return _IndexedSequence(self, indices)

    def __iter__(self):
        """Iterate over frames of the Sequence.

        Yields numpy arrays of shape (num_planes, num_rows, num_columns,
        num_channels).
        """
        for t in range(len(self)):
            yield self._get_frame(t)

    @abstractmethod
    def _get_frame(self, n):
        """Get the nth frame.

        Parameters
        ----------
        n : int
            The index of the frame.

        Returns
        -------
        frame : np.ndarray
            Shape: (num_planes, num_rows, num_columns, num_channels)
        """
        raise NotImplementedError

    def __len__(self):
        """The number of frames in the sequence."""
        return sum(1 for _ in self)

    @property
    def shape(self):
        """Return shape of the sequence."""
        return (len(self),) + self._get_frame(0).shape

    def apply_displacements(self, displacements, frame_shape=None):
        """Wrap the sequence to apply motion correction displacements."""
        return _MotionCorrectedSequence(self, displacements, frame_shape)

    @staticmethod
    def join(*sequences):
        """Join sequences representing different channels.

        Parameters
        ----------
        sequences : Sequence
            Each argument is a Sequence representing a different channel.

        Returns
        -------
        joined_sequence : Sequence
            A single sequence with multiple channels.
        """
        return _JoinedSequence(sequences)


class ArraySequence(Sequence):
    """A Sequence backed by a numpy array.

    Parameters
    ----------
    array : np.ndarray
        5D array with shape (num_frames, num_planes, num_rows, num_columns,
        num_channels).
    """

    def __init__(self, array):
        self._array = np.asarray(array)
        if self._array.ndim != 5:
            raise ValueError(
                f'Expected 5D array, got {self._array.ndim}D')

    def _get_frame(self, n):
        return self._array[n]

    def __len__(self):
        return self._array.shape[0]

    @property
    def shape(self):
        return self._array.shape


class _JoinedSequence(Sequence):
    """Sequence joining multiple single-channel sequences."""

    def __init__(self, sequences):
        shape = None
        num_channels = 0
        self._sequences = sequences
        for seq in sequences:
            if shape is None:
                shape = seq.shape[:-1]
            if not shape == seq.shape[:-1]:
                raise ValueError(
                    'Sequences being joined must have the same number '
                    'of frames, planes, rows, and columns.')
            num_channels += seq.shape[-1]
        self._shape = shape + (num_channels,)

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        return self._shape

    def __iter__(self):
        for frames in zip(*self._sequences):
            yield np.concatenate(frames, axis=3)

    def _get_frame(self, t):
        return np.concatenate(
            [seq._get_frame(t) for seq in self._sequences], axis=3)


class _WrapperSequence(Sequence, metaclass=ABCMeta):
    """Abstract class for wrapping a Sequence to modify its functionality."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)


class _MotionCorrectedSequence(_WrapperSequence):
    """Wraps a sequence to apply motion correction.

    Parameters
    ----------
    base : Sequence
    displacements : array
        The displacement of each row in the image cycle.
        Shape: (num_frames, num_planes, num_rows, displacement_dim).
    extent : tuple, optional
        (num_planes, num_rows, num_columns)
    """

    def __init__(self, base, displacements, extent=None):
        super().__init__(base)
        if np.min(displacements) < 0:
            raise ValueError("All displacements must be non-negative")
        self.displacements = displacements.astype(int)
        if extent is None:
            max_disp = np.nanmax(
                [np.nanmax(d.reshape(-1, d.shape[-1]), 0)
                 for d in displacements], 0)
            extent = np.array(base.shape)[1:-1]
            extent[1:3] += max_disp
        assert len(extent) == 3
        self._frame_shape_zyx = tuple(extent)

    @property
    def _frame_shape(self):
        return self._frame_shape_zyx + (self._base.shape[4],)

    def __len__(self):
        return len(self._base)

    def _align(self, frame, displacement):
        if displacement.ndim == 3:
            return _align_frame(
                frame.astype(float), displacement.astype(int),
                self._frame_shape)
        elif displacement.ndim == 2:
            out = np.nan * np.ones(self._frame_shape)
            s = frame.shape
            for p, (plane, disp) in enumerate(zip(frame, displacement)):
                if len(disp) == 2:
                    disp = [0] + list(disp)
                out[p + disp[0],
                    disp[1]:(disp[1] + s[1]),
                    disp[2]:(disp[2] + s[2])] = plane
            return out
        elif displacement.ndim == 1:
            out = np.nan * np.ones(self._frame_shape)
            s = frame.shape
            out[displacement[0]:(displacement[0] + s[0]),
                displacement[1]:(displacement[1] + s[1]),
                displacement[2]:(displacement[2] + s[2])] = frame
            return out

    @property
    def shape(self):
        return (len(self),) + self._frame_shape

    def __iter__(self):
        for frame, displacement in zip(self._base, self.displacements):
            yield self._align(frame, displacement)

    def _get_frame(self, t):
        return self._align(self._base._get_frame(t), self.displacements[t])

    def __getitem__(self, indices):
        if len(indices) > 5:
            raise ValueError
        indices = indices if isinstance(indices, tuple) else (indices,)
        times = indices[0]
        if indices[0] not in (None, slice(None)):
            new_indices = (slice(None),) + indices[1:]
            return _MotionCorrectedSequence(
                self._base[times],
                self.displacements[times],
                self._frame_shape[:-1]
            )[new_indices]
        if len(indices) == 5:
            chans = indices[4]
            return _MotionCorrectedSequence(
                self._base[:, :, :, :, chans],
                self.displacements,
                self._frame_shape[:-1]
            )[indices[:4]]
        return _IndexedSequence(self, indices)


class _IndexedSequence(_WrapperSequence):
    """Sequence created by indexing/slicing another Sequence."""

    def __init__(self, base, indices):
        super().__init__(base)
        self._base_len = len(base)
        self._indices = \
            indices if isinstance(indices, tuple) else (indices,)
        new_indices = []
        for i in self._indices:
            try:
                i = int(i)
            except TypeError:
                new_indices.append(i)
            else:
                new_indices.append(slice(i, i + 1))
        self._indices = tuple(new_indices)
        self._times = list(range(self._base_len))[self._indices[0]]

    def __iter__(self):
        try:
            for t in self._times:
                yield np.copy(self._base._get_frame(t)[self._indices[1:]])
        except NotImplementedError:
            if self._indices[0].step is not None and self._indices[0].step < 0:
                raise NotImplementedError(
                    'Iterating backwards not supported by the base class')
            idx = 0
            for t, frame in enumerate(self._base):
                try:
                    whether_yield = t == self._times[idx]
                except IndexError:
                    return
                if whether_yield:
                    yield np.copy(frame[self._indices[1:]])
                    idx += 1

    def _get_frame(self, t):
        return self._base._get_frame(self._times[t])[self._indices[1:]]

    def __len__(self):
        return len(list(range(len(self._base)))[self._indices[0]])


class ImagingDataset:
    """A multiple sequence imaging dataset.

    Parameters
    ----------
    sequences : list of Sequence
        Imaging sequences.
    savedir : str or None
        The directory used to store the dataset. None for in-memory only.
    channel_names : list of str, optional
        Names for the channels.
    """

    def __init__(self, sequences, savedir=None, channel_names=None,
                 read_only=False):
        if sequences is None:
            raise ValueError('Cannot initialize dataset without sequences.')
        if not all(isinstance(s, Sequence) for s in sequences):
            raise TypeError(
                'ImagingDataset must be initialized with a list of Sequences.')
        self.sequences = list(sequences)
        self.savedir = savedir
        if channel_names is None:
            self.channel_names = [
                str(x) for x in range(self.frame_shape[-1])]
        else:
            self.channel_names = channel_names

    def __iter__(self):
        """Iterate over sequences."""
        return iter(self.sequences)

    @property
    def frame_shape(self):
        """Shape of each frame: (num_planes, num_rows, num_columns, num_channels)."""
        return self.sequences[0].shape[1:]

    @property
    def num_sequences(self):
        return len(self.sequences)

    @property
    def num_frames(self):
        return sum(len(s) for s in self.sequences)
