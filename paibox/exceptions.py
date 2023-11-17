class PAIBoxError(Exception):
    """General exception for PAIBox"""

    pass


class PAIBoxWarning(Warning):
    """General warning for PAIBox"""

    pass


class ShapeError(PAIBoxError):
    """Exception for incorrect shape"""

    pass


class RegisterError(PAIBoxError):
    """Raise when registering an object fails"""

    pass


class BuildError(PAIBoxError):
    """Raise when building fails."""

    pass


class NotSupportedError(PAIBoxError, NotImplementedError):
    """Exception for a certain function not supported."""

    pass


class SimulationError(PAIBoxError, RuntimeError):
    """An error encountered during simulation."""

    pass


class PAICoreResourceError(PAIBoxError):
    pass
