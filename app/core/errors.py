"""Exception hierarchy of the package."""


class PhotoCheckError(Exception):
    """Base class for all package errors."""


class SpecError(PhotoCheckError):
    """Raised when a requirement specification is malformed."""


class UnknownMeasurerError(SpecError):
    """Raised when a specification references a measurer absent from the registry."""


class ModelMissingError(PhotoCheckError):
    """Raised when a required model file has not been downloaded."""


class ImageReadError(PhotoCheckError):
    """Raised when an image file cannot be read or decoded."""


class LlmError(PhotoCheckError):
    """Raised when a language model request fails."""
