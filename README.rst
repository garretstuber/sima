SIMA — Sequential IMage Analysis (Modernized Fork)
====================================================

Overview
--------

This repository is a **modernized fork** of
`SIMA <https://github.com/losonczylab/sima>`_
(Sequential IMage Analysis), an open source Python package originally
developed by the `Losonczy Lab <http://losonczylab.org>`_ at Columbia
University for analysis of time-series fluorescence imaging data.

The original SIMA package (Kaifosh et al., 2014) targeted Python 2.7 and
relied on deprecated dependencies that prevent it from running on modern
Python environments.  This fork extracts and modernizes the
**motion correction** algorithms — the most valuable part of SIMA — into
a new standalone package called ``sima_motion``.

What is ``sima_motion``?
------------------------

``sima_motion`` is a self-contained Python 3.10+ package providing three
families of motion correction algorithms for laser-scanning fluorescence
microscopy data:

* **Hidden Markov Model (HMM)** — Line-by-line Viterbi-based correction
  that fixes *within-frame* motion artifacts caused by animal movement
  during scanning.  This is the algorithm that set SIMA apart from other
  tools.
* **Discrete Fourier Transform (DFT) registration** — Efficient
  subpixel image registration by cross-correlation, with iterative
  mean-image refinement.
* **Frame alignment** — Pyramid-based cross-correlation alignment for
  2-D planes and 3-D volumes.

All three methods work on multi-plane, multi-channel imaging data and
support parallel processing.

What changed from the original SIMA?
--------------------------------------

Only the motion correction module was extracted.  Segmentation, ROI
handling, signal extraction, and the ROI Buddy GUI are **not** included.

The following modernizations were applied to every file:

* Removed all Python 2 compatibility layers (``future``, ``builtins``,
  ``past``, ``old_div``, ``with_metaclass``).
* Replaced deprecated NumPy aliases (``np.int`` → ``np.intp``,
  ``np.float`` → ``np.float64``, ``np.product`` → ``np.prod``).
* Fixed removed SciPy import paths
  (``scipy.ndimage.filters`` → ``scipy.ndimage``,
  ``scipy.ndimage.interpolation.shift`` → ``scipy.ndimage.shift``,
  ``scipy.stats.mstats.mquantiles`` → ``np.quantile``).
* Modernized the Cython extension (``_motion.pyx``) for NumPy ≥ 1.25.
* Created minimal ``Sequence`` / ``ImagingDataset`` abstractions
  decoupled from the full SIMA package, plus a convenient
  ``ArraySequence`` wrapper for plain NumPy arrays.
* Switched to ``pyproject.toml``-based packaging.

Requirements
------------

**Required (runtime)**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - `Python <https://python.org>`_
     - >= 3.10
     - Language runtime
   * - `NumPy <https://numpy.org>`_
     - >= 1.25
     - Array operations
   * - `SciPy <https://scipy.org>`_
     - >= 1.10
     - Scientific computing (ndimage, fftpack, special, stats, sparse)

**Required (build only)**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - `Cython <https://cython.org>`_
     - >= 3.0
     - Compiles the ``_motion.pyx`` extension
   * - `setuptools <https://setuptools.pypa.io>`_
     - >= 68
     - Build backend

**Optional (faster performance)**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - `pyFFTW <https://pyfftw.readthedocs.io>`_
     - >= 0.13
     - Faster FFT via FFTW (used in cross-correlation and DFT registration)
   * - `Bottleneck <https://bottleneck.readthedocs.io>`_
     - >= 1.3
     - Faster NaN-aware reductions (nanmean, nansum, nanmedian)

**Optional (development)**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - `pytest <https://pytest.org>`_
     - >= 7
     - Running tests

Installation
------------

.. code-block:: bash

   cd sima_motion
   pip install -e .

To install with optional fast FFT and development dependencies:

.. code-block:: bash

   pip install -e ".[fast,dev]"

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from sima_motion import PlaneTranslation2D, ArraySequence, ImagingDataset

   # Load your imaging data as a 5-D array:
   # (num_frames, num_planes, num_rows, num_columns, num_channels)
   data = np.load("my_imaging_data.npy")

   # Wrap in sima_motion data structures
   seq = ArraySequence(data)
   dataset = ImagingDataset([seq])

   # Estimate motion (frame-level 2-D translation)
   strategy = PlaneTranslation2D(max_displacement=[20, 20])
   displacements = strategy.estimate(dataset)

   # Or use the HMM for line-by-line correction
   from sima_motion import HiddenMarkov2D

   hmm = HiddenMarkov2D(
       granularity='row',
       max_displacement=[30, 30],
       verbose=True,
   )
   displacements = hmm.estimate(dataset)

   # Apply displacements to get corrected data
   corrected = dataset.sequences[0].apply_displacements(
       displacements[0], frame_shape=None
   )

Package Structure
-----------------

::

   sima_motion/
   ├── pyproject.toml        # Build configuration
   ├── setup.py              # Cython extension build
   └── sima_motion/
       ├── __init__.py        # Public API
       ├── hmm.py             # HMM motion correction (Kaifosh et al. 2013)
       ├── dftreg.py          # DFT subpixel registration
       ├── frame_align.py     # Pyramid cross-correlation alignment
       ├── motion.py          # Base strategy + resonant scanner correction
       ├── _motion.pyx        # Cython-optimized HMM routines
       ├── align.py           # Cross-correlation utilities
       ├── sequence.py        # Sequence / ImagingDataset abstractions
       ├── transform.py       # Abstract transform classes
       └── tests/
           ├── test_frame_align.py
           └── test_dftreg.py

Running Tests
-------------

.. code-block:: bash

   cd sima_motion
   pip install -e ".[dev]"
   pytest sima_motion/tests/ -v

Citing SIMA
-----------

If you use this software, please cite the original SIMA paper:

   Kaifosh P, Zaremba JD, Danielson NB, and Losonczy A. SIMA: Python
   software for analysis of dynamic fluorescence imaging data.
   *Frontiers in Neuroinformatics*. 2014; 8:80.
   `doi:10.3389/fninf.2014.00080 <https://doi.org/10.3389/fninf.2014.00080>`_

License
-------

Unless otherwise specified in individual files, all code is

Copyright (C) 2014 The Trustees of Columbia University in the City of
New York.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
