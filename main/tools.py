_PLIB_BASE_INTRO = """
To install or update to the latest version, pip install paicorelib.

To use the development version, pip install --pre paicorelib."""

PLIB_INSTALL_INTRO = (
    "\nPAIBox requires paicorelib, please install it.\n" + _PLIB_BASE_INTRO
)

PLIB_UPDATE_INTRO = (
    "\nThe minimum required version of paicorelib is {0}, but the current version is {1}.\n"
    + _PLIB_BASE_INTRO
)
