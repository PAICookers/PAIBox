import warnings

try:
    import paicorelib
except ImportError:
    raise ImportError(
        "The runtime requires paicorelib. Please install it by running 'pip install paicorelib'."
    ) from None

del paicorelib

# Version check for standablone scenario
MAX_PLIB_VERSION = "1.5.0"
from paicorelib import __version__ as plib_version

if plib_version is not None:
    if plib_version >= MAX_PLIB_VERSION:
        raise ImportError(
            f"The runtime requires paicorelib version < {MAX_PLIB_VERSION}, but {plib_version} is installed."
        ) from None
else:
    warnings.warn(
        f"No exact version found, make sure the paicorelib version < {MAX_PLIB_VERSION}."
    )

from .runtime import PAIBoxRuntime
