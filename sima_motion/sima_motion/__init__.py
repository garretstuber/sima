"""sima_motion: Motion correction for fluorescence imaging data.

Extracted from SIMA (Kaifosh et al., 2014, Front. Neuroinform.).

Provides line-by-line HMM-based motion correction for within-frame
motion artifacts during laser scanning microscopy, as well as
DFT-based registration and frame alignment methods.
"""

from .motion import MotionEstimationStrategy, ResonantCorrection
from .frame_align import PlaneTranslation2D, VolumeTranslation
from .hmm import HiddenMarkov2D, HiddenMarkov3D, MovementModel
from .dftreg import DiscreteFourier2D
from .sequence import Sequence, ArraySequence, ImagingDataset

__version__ = "1.0.0"

__all__ = [
    "MotionEstimationStrategy",
    "ResonantCorrection",
    "PlaneTranslation2D",
    "VolumeTranslation",
    "HiddenMarkov2D",
    "HiddenMarkov3D",
    "MovementModel",
    "DiscreteFourier2D",
    "Sequence",
    "ArraySequence",
    "ImagingDataset",
]
