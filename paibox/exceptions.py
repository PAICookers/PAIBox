class PAIBoxError(Exception):
    """General exception for PAIBox."""

    pass


class PAIBoxWarning(Warning):
    """General warning for PAIBox."""

    pass


class ParameterInvalidWarning(PAIBoxWarning):
    """Parameter is invalid due to some reason."""

    pass


class ShapeError(PAIBoxError):
    """Exception for incorrect shape."""

    pass


class RegisterError(PAIBoxError, KeyError):
    """Raise when registering an object fails."""

    pass


class BuildError(PAIBoxError, RuntimeError):
    """Raise when building fails."""

    pass


class NotSupportedError(PAIBoxError, NotImplementedError):
    """Exception for a certain function not supported."""

    pass


class SimulationError(PAIBoxError, RuntimeError):
    """An error encountered during simulation."""

    pass


class ResourceError(PAIBoxError):
    """Resource usage exceeds hardware limit."""

    pass


class FrameIllegalError(PAIBoxError, ValueError):
    """Frame is illegal."""

    pass


class TruncationWarning(PAIBoxWarning, UserWarning):
    """Value out of range & will be truncated."""

    pass
