from typing import final


class PAIBoxError(Exception):
    """General exception for PAIBox."""

    pass


class PAIBoxWarning(UserWarning):
    """General warning for PAIBox."""

    pass


class PAIBoxDeprecationWarning(PAIBoxWarning, DeprecationWarning):
    """Warning class for features which will be deprecatedin a future version."""

    pass


class ConfigInvalidError(PAIBoxError, ValueError):
    """Configuration is invalid."""

    pass


class ParameterInvalidWarning(PAIBoxWarning):
    """Parameter is invalid due to some reason."""

    pass


@final
class ShapeError(PAIBoxError):
    """Exception for incorrect shape."""

    pass


class RegisterError(PAIBoxError, KeyError):
    """Raise when registering an object fails."""

    pass


class GraphBuildError(PAIBoxError):
    """Raise when building PAIGraph fails."""

    pass


class GraphConnectionError(GraphBuildError):
    """Connection error."""

    pass


class NotSupportedError(PAIBoxError, NotImplementedError):
    """Exception for a certain function not supported."""

    pass


class SimulationError(PAIBoxError, RuntimeError):
    """An error encountered during simulation."""

    pass


class FunctionalError(PAIBoxError, RuntimeError):
    """Functional errors, usually hardware related register."""

    pass


class RoutingError(PAIBoxError):
    """Exception for routing tree."""

    pass


class ResourceError(PAIBoxError):
    """Resource usage exceeds hardware limit."""

    pass


class FrameIllegalError(PAIBoxError, ValueError):
    """Frame is illegal."""

    pass


class TruncationWarning(PAIBoxWarning):
    """Value out of range & will be truncated."""

    pass


class AutoOptimizationWarning(PAIBoxWarning):
    """Parameters are optimized automatically by PAIBox."""

    pass
