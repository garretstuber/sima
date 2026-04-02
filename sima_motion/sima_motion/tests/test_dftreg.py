"""Tests for DFT registration motion correction."""

import numpy as np
import pytest

from sima_motion.dftreg import _register_translation, DiscreteFourier2D
from sima_motion.sequence import ArraySequence, ImagingDataset


def test_register_translation():
    """Test that DFT registration recovers known shifts."""
    np.random.seed(0)
    reference = np.random.rand(64, 64)
    target = np.zeros_like(reference)
    dy, dx = 3, 5
    target[dy:, dx:] = reference[:64 - dy, :64 - dx]
    shifts = _register_translation(reference, target)
    np.testing.assert_array_almost_equal(shifts, [dy, dx], decimal=0)


def test_discrete_fourier_2d():
    """Test DiscreteFourier2D with synthetic data."""
    np.random.seed(42)
    num_frames = 5
    rows, cols = 64, 64
    base_image = np.random.rand(rows, cols)

    frames = np.zeros((num_frames, 1, rows, cols, 1))
    shifts_applied = []
    for i in range(num_frames):
        dy = np.random.randint(-5, 6)
        dx = np.random.randint(-5, 6)
        shifts_applied.append((dy, dx))
        shifted = np.zeros((rows, cols))
        sy = max(0, dy)
        sx = max(0, dx)
        ey = rows - max(0, -dy)
        ex = cols - max(0, -dx)
        shifted[sy:ey, sx:ex] = base_image[
            max(0, -dy):rows - max(0, dy),
            max(0, -dx):cols - max(0, dx)]
        frames[i, 0, :, :, 0] = shifted

    seq = ArraySequence(frames)
    dataset = ImagingDataset([seq])
    strategy = DiscreteFourier2D(max_displacement=[10, 10], verbose=False)
    displacements = strategy.estimate(dataset)

    assert len(displacements) == 1
    assert displacements[0].shape[0] == num_frames
