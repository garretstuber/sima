"""Tests for frame alignment motion correction."""

import numpy as np
import pytest

from sima_motion.frame_align import (
    shifted_corr, base_alignment, pyramid_align, _resize_array,
    _update_sums_and_counts, PlaneTranslation2D
)
from sima_motion.sequence import ArraySequence, ImagingDataset


def test_shifted_corr():
    """Test that shifted_corr returns high correlation for identical images."""
    np.random.seed(0)
    im = np.random.rand(5, 10, 10, 2)
    corr = shifted_corr(im, im, np.array([0, 0, 0]))
    assert corr > 0.99


def test_base_alignment():
    """Test that base_alignment recovers a known shift."""
    np.random.seed(0)
    reference = np.random.rand(10, 10, 1)
    target = np.zeros_like(reference)
    target[2:, 3:, :] = reference[:-2, :-3, :]
    shift = base_alignment(reference, target)
    np.testing.assert_array_equal(shift, [2, 3])


def test_pyramid_align():
    """Test that pyramid_align recovers a known shift."""
    np.random.seed(0)
    reference = np.random.rand(64, 64, 1)
    target = np.zeros_like(reference)
    target[3:, 5:, :] = reference[:-3, :-5, :]
    shift = pyramid_align(reference, target)
    np.testing.assert_array_equal(shift, [3, 5])


def test_resize_array():
    """Test array resizing with displacement."""
    a = np.ones((2, 128, 128, 5))
    a = _resize_array(a, (0, -1, 2), (2, 128, 128, 5))
    assert a.shape == (2, 129, 130, 5)


def test_update_sums_and_counts():
    """Test update of reference image sums and counts."""
    pixel_counts = np.zeros((4, 5, 5, 2))
    pixel_sums = np.zeros((4, 5, 5, 2))
    plane = 2 * np.ones((1, 2, 3, 2))
    pixel_sums, pixel_counts = _update_sums_and_counts(
        pixel_sums, pixel_counts, [0, 0, 0], [3, 1, 2], plane)
    assert np.all(pixel_sums[3, 1:3, 2:5] == 2)
    assert np.all(pixel_counts[3, 1:3, 2:5] == 1)


def test_plane_translation_2d():
    """Test PlaneTranslation2D with synthetic shifted data."""
    np.random.seed(42)
    num_frames = 10
    rows, cols = 64, 64
    base_image = np.random.rand(rows, cols)

    # Create shifted frames (all shifted by [3, 5])
    frames = np.zeros((num_frames, 1, rows, cols, 1))
    for i in range(num_frames):
        dy, dx = 3, 5
        shifted = np.zeros((rows, cols))
        shifted[dy:, dx:] = base_image[:rows - dy, :cols - dx]
        frames[i, 0, :, :, 0] = shifted

    seq = ArraySequence(frames)
    dataset = ImagingDataset([seq])
    strategy = PlaneTranslation2D(max_displacement=[10, 10])
    displacements = strategy.estimate(dataset)

    # All frames should have similar displacements
    assert len(displacements) == 1
    assert displacements[0].shape[0] == num_frames
