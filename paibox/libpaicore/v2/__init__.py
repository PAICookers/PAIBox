from .coordinate import *
from .frame_defs import *
from .hw_defs import HwConfig as HwConfig
from .hw_types import *
from .ram_model import *
from .ram_types import LeakingComparisonMode as LCM
from .ram_types import LeakingDirectionMode as LDM
from .ram_types import LeakingIntegrationMode as LIM
from .ram_types import NegativeThresholdMode as NTM
from .ram_types import ResetMode as RM
from .ram_types import SynapticIntegrationMode as SIM
from .ram_types import ThresholdMode as TM
from .reg_model import *
from .reg_types import CoreMode as CoreMode
from .reg_types import CoreModeDict as CoreModeDict
from .reg_types import InputWidthFormatType as InputWidthFormat
from .reg_types import LCNExtensionType as LCN_EX
from .reg_types import MaxPoolingEnableType as MaxPoolingEnable
from .reg_types import SNNModeEnableType as SNNModeEnable
from .reg_types import SpikeWidthFormatType as SpikeWidthFormat
from .reg_types import WeightPrecisionType as WeightPrecision
from .routing_defs import *

__major__ = 2
__minor__ = 0
__revision__ = 0
__version__ = f"{__major__}.{__minor__}.{__revision__}"


def get_version_json():
    return {
        "major": __major__,
        "minor": __minor__,
        "revision": __revision__,
        "version": __version__,
    }
