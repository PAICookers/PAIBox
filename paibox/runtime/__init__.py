import warnings

try:
    import paicorelib
except ImportError:
    raise ImportError(
        "The runtime requires paicorelib. Please install it by running 'pip install paicorelib'."
    ) from None

del paicorelib

# Version check for standablone scenario
# In case of breaking changes in paicorelib, maximum allowed version is set here.
MAX_SUPPORT_PLIB_VER = "1.5.0"
from paicorelib import __version__ as plib_ver
from packaging import version as pkg_version

if plib_ver is not None:
    if pkg_version.parse(plib_ver) >= pkg_version.parse(MAX_SUPPORT_PLIB_VER):
        raise ImportError(
            f"The runtime only support paicorelib version < {MAX_SUPPORT_PLIB_VER}, but {plib_ver} is installed."
        ) from None
else:
    warnings.warn(
        f"No exact version found, make sure the paicorelib version < {MAX_SUPPORT_PLIB_VER}."
    )

from .runtime import PAIBoxRuntime
