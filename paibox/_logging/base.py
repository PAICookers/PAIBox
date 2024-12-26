import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict, Union
from weakref import WeakSet

from paibox.exceptions import RegisterError

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

# Refer to torch/_logging module: https://github.com/pytorch/pytorch/blob/main/torch/_logging.

DEFAULT_LOG_LEVEL = logging.WARNING
LOG_ENV_VAR = "PB_LOGS"
LOG_OUT_ENV_VAR = "PB_LOGS_OUT"
LOG_FMT_ENV_VAR = "PB_LOGS_FMT"


log = logging.getLogger(__name__)


@dataclass
class LogRegistry:
    log_alias_to_log_names: dict[str, list[str]] = field(default_factory=dict)
    """Shorthand name to log name."""

    artifact_log_names: set[str] = field(default_factory=set)
    """Artifact logger names. Currently formatted as <module>.__<artifact_name>."""

    artifact_desc: dict[str, str] = field(default_factory=dict)
    """A short description of each artifact."""

    off_by_default_artifact_names: set[str] = field(default_factory=set)
    """Artifacts which are not displayed unless named in the settings explicitly."""

    artifact_names: set[str] = field(default_factory=set)
    """Artifact names, populated by `register_artifact()`."""

    def is_artifact(self, artf: str) -> bool:
        return artf in self.artifact_names

    def is_log(self, alias: str) -> bool:
        return alias in self.log_alias_to_log_names

    def _register_artifact_name(
        self, name: str, desc: Optional[str], off_by_default: bool
    ) -> None:
        self.artifact_names.add(name)
        self.artifact_desc[name] = desc if desc is not None else ""

        # if off by default, don't enable it when log_name's log_level is set to DEBUG
        if off_by_default:
            self.off_by_default_artifact_names.add(name)

    def _register_log(self, alias: str, log_names: Union[str, list[str]]) -> None:
        if isinstance(log_names, str):
            log_names = [log_names]

        self.log_alias_to_log_names[alias] = log_names

    def _register_artifact_log(self, artifact_log_name: str) -> None:
        self.artifact_log_names.add(artifact_log_name)

    def get_log_names(self) -> set[str]:
        return {
            name for names in self.log_alias_to_log_names.values() for name in names
        }

    def get_artifact_log_names(self) -> set[str]:
        return set(self.artifact_log_names)

    def is_off_by_default(self, arft_name: str) -> bool:
        return arft_name in self.off_by_default_artifact_names


@dataclass
class LogState:
    log_name_to_level: dict[str, int] = field(default_factory=dict)
    artifact_names: set[str] = field(default_factory=set)
    """The set of currently enabled artifacts."""

    def enable_artifact(self, artf: str) -> None:
        self.artifact_names.add(artf)

    def is_artifact_enabled(self, name: str) -> bool:
        return name in self.artifact_names

    def enable_log(self, log_name: Union[str, list[str]], log_level: int) -> None:
        if isinstance(log_name, str):
            log_name = [log_name]

        for log_name in log_name:
            self.log_name_to_level[log_name] = log_level

    def get_log_level_pairs(self):
        return self.log_name_to_level.items()

    def clear(self) -> None:
        self.log_name_to_level.clear()
        self.artifact_names.clear()


log_registry = LogRegistry()
log_state = LogState()


def register_log(short_name: str, log_names: Union[str, list[str]]) -> None:
    log_registry._register_log(short_name, log_names)


def register_artifact(
    setting_name: str, desc: Optional[str] = None, off_by_default: bool = True
) -> None:
    log_registry._register_artifact_name(setting_name, desc, off_by_default)


def get_artifact_logger(module_name: str, artifact_name: str) -> logging.Logger:
    if artifact_name not in log_registry.artifact_names:
        raise RegisterError(
            f"artifact name: {repr(artifact_name)} not registered, "
            f"please call register_artifact({repr(artifact_name)}) in paibox._logging.registrations."
        )

    name = module_name + f".__{artifact_name}"
    log = logging.getLogger(name)
    log.artifact_name = artifact_name  # type: ignore[attr-defined]
    log_registry._register_artifact_log(name)
    _configure_artifact_log(log)

    return log


def _configure_artifact_log(log: logging.Logger) -> None:
    # If the artifact is off by default, then it should only be logged when explicitly
    # enabled; set propagate to False so that this artifact is not propagated
    # to its ancestor logger
    if log_registry.is_off_by_default(log.artifact_name):  # type: ignore[attr-defined]
        log.propagate = False

    # Enable artifact logging when explicitly enabled
    if log_state.is_artifact_enabled(log.artifact_name):  # type: ignore[attr-defined]
        log.setLevel(logging.DEBUG)
        log.propagate = True


log_handlers = WeakSet()


log_level_to_abbr = {
    "DEBUG": "D",
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C",
}


class PAIBoxLogsFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        artifact_name = getattr(logging.getLogger(record.name), "artifact_name", None)
        record.artifactprefix = ""
        if artifact_name is not None:
            record.artifactprefix = f" [__{artifact_name}]"

        s = super().format(record)
        record.asctime = self.formatTime(record, "%H:%M:%S")

        shortlevel = log_level_to_abbr.get(record.levelname, record.levelname)
        prefix = f"[{shortlevel} {record.asctime} {record.lineno}]{record.artifactprefix}"  # type: ignore[attr-defined]

        lines = s.split("\n")
        return "\n".join(f"{prefix} {l}" for l in lines)


def _default_formatter() -> logging.Formatter:
    if (fmt := os.environ.get(LOG_FMT_ENV_VAR, None)) is None:
        return PAIBoxLogsFormatter()
    else:
        if fmt.lower() in ("default", "short", "basic"):
            fmt = logging.BASIC_FORMAT

        return logging.Formatter(fmt)


DEFAULT_FORMATTER = _default_formatter()


def _setup_handlers(
    create_handler_fn: Callable[..., logging.Handler], log: logging.Logger
) -> None:
    hdlr = _track_handler(create_handler_fn())
    hdlr.setFormatter(DEFAULT_FORMATTER)
    hdlr.setLevel(logging.DEBUG)
    log.addHandler(hdlr)


def _track_handler(handler: logging.Handler) -> logging.Handler:
    log_handlers.add(handler)
    return handler


def _clear_handlers(log: logging.Logger) -> None:
    # clears all handlers on specified loggers
    to_remove = [handler for handler in log.handlers if handler in log_handlers]
    for handler in to_remove:
        log.removeHandler(handler)


def _reset_logs() -> None:
    """Reset all registered logs."""
    for log_name in log_registry.get_log_names():
        log = logging.getLogger(log_name)
        log.setLevel(DEFAULT_LOG_LEVEL)
        log.propagate = False
        _clear_handlers(log)


def _get_log_state() -> LogState:
    return log_state


def _set_log_state(state) -> None:
    global log_state
    log_state = state


def _init_logs(log_file_name: Optional[Union[str, Path]] = None) -> None:
    _reset_logs()
    # _update_log_state_from_env()

    if (out := os.environ.get(LOG_OUT_ENV_VAR, None)) is not None:
        log_file_name = out

    # 1. Reset all registered loggers to NOTSET, so that they respect their parent log level.
    for log_name in log_registry.get_log_names():
        # Except the top level
        if log_name == "paibox":
            continue

        log = logging.getLogger(log_name)
        log.setLevel(logging.NOTSET)

    # 2. For all loggers which the user requested to have non-standard logging behavior,
    # modify their log levels.
    for log_name, level in log_state.get_log_level_pairs():
        log = logging.getLogger(log_name)
        log.setLevel(level)

    # 3. Setup handlers for all registered loggers
    for log_name in log_registry.get_log_names():
        log = logging.getLogger(log_name)
        _setup_handlers(logging.StreamHandler, log)

        if log_file_name is not None:
            _setup_handlers(lambda: logging.FileHandler(log_file_name), log)

    # 4. Configure artifact loggers.
    # NOTE: this must happen last since the levels of ancestor loggers are taken into account.
    for artifact_log_qname in log_registry.get_artifact_log_names():
        log = logging.getLogger(artifact_log_qname)
        _configure_artifact_log(log)


# Add it to this dictionary if registering a new log in `registrations.py`.
class _LogSettingsKwds(TypedDict, total=False):
    paibox: Optional[int]
    backend: Optional[int]
    sim: Optional[int]
    build_core_blocks: bool
    lcn_ex_adjustment: bool
    cb_axon_grouping: bool
    coord_assign: bool
    get_dest: bool
    routing_group_info: bool


# Add a default log level or state for each log or artifact name in the above dictionary.
DEFAULT_LOG_SETTINGS: _LogSettingsKwds = {
    "paibox": logging.INFO,
    "backend": logging.INFO,
    "sim": logging.INFO,
    "build_core_blocks": True,
    "lcn_ex_adjustment": True,
    "cb_axon_grouping": True,
    "coord_assign": True,
    "get_dest": True,
    "routing_group_info": True,
}


def set_logs(**kwargs: Unpack[_LogSettingsKwds]) -> None:
    """Set the log levels for registered logs and artifacts.

    Example:
    >>> set_logs(paibox=logging.INFO, backend=logging.DEBUG)
    >>> set_logs(**DEFAULT_LOG_SETTINGS)
    """
    global log_state
    log_state.clear()

    for alias, val in kwargs.items():
        if val is None:
            continue

        if log_registry.is_artifact(alias):
            if not isinstance(val, bool):
                raise ValueError(
                    f"expected bool to enable artifact {alias}, but got {val}."
                )

            if val:
                log_state.enable_artifact(alias)
        elif log_registry.is_log(alias):
            if val not in logging._levelToName:
                raise ValueError(
                    f"unrecognized log level for log {alias}: {val}, valid level values "
                    f"are: {','.join([str(k) for k in logging._levelToName.keys()])}."
                )

            log_state.enable_log(
                log_registry.log_alias_to_log_names.get(alias, alias), val
            )
        else:
            raise ValueError(
                f"unrecognized log or artifact name passed to set_logs: {alias}."
            )

        _init_logs()


class LazyString:
    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.func(*self.args, **self.kwargs)
