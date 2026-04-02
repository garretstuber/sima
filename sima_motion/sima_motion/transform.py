"""Abstract geometric transform classes."""

import abc


class Transform(metaclass=abc.ABCMeta):
    """Abstract class for geometric transforms."""

    @abc.abstractmethod
    def apply(self, source, grid=None):
        """Apply the transform to raw source data.

        Parameters
        ----------
        source : np.ndarray
        grid : optional

        Returns
        -------
        transformed : np.ndarray
        """
        pass


class InvertibleTransform(Transform, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def inverse(self):
        pass


class DifferentiableTransform(Transform, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def jacobian(self):
        pass


class NullTransform(Transform):
    """Represents a null transform (e.g. when estimation fails)."""

    def apply(self, source, grid=None):
        return source


class Identity(Transform):

    def apply(self, source, grid=None):
        return source


class WithinFrameTranslation(Transform):

    def apply(self, source, grid=None):
        return source
