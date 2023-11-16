class ShapeError(Exception):
    """Exception for incorrect shape"""

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class IndexProbeError(IndexError):
    """Exception for incorrect Probe index"""

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class RegisterError(LookupError):
    """When a mapping or sequence does not allow registering an already existing key or index, an exception is
    raised."""

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class StatusError(Exception):
    """Exception for not executing some prerequisites"""

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class PAICoreError(Exception):
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass
