import logging

import pytest

from paibox import _logging
from paibox._logging.base import DEFAULT_LOG_SETTINGS


def _set_log_state_from_mark(request):
    mark = request.node.get_closest_marker("make_settings_test")
    if mark:
        if mark.kwargs:
            _logging.base.set_logs(**mark.kwargs)
        else:
            _logging.base.set_logs(**DEFAULT_LOG_SETTINGS)

    return mark


@pytest.fixture
def log_settings_patch(request):
    prev = _logging.base._get_log_state()
    _logging.base._set_log_state(_logging.base.LogState())

    mark = _set_log_state_from_mark(request)

    yield
    _logging.base._set_log_state(prev)
    _logging._init_logs()


@pytest.fixture
def captured_logs(monkeypatch):
    captured: list[logging.LogRecord] = []

    for log_name in _logging.base.log_registry.get_log_names():
        logger = logging.getLogger(log_name)
        assert len(logger.handlers) > 0

        for handler in logger.handlers:
            old_emit = handler.emit

            def new_emit(record) -> None:
                old_emit(record)
                captured.append(record)

            monkeypatch.setattr(handler, "emit", new_emit)

    yield captured


def has_record(records: list[logging.LogRecord], kwd: str) -> bool:
    return any(kwd in r.getMessage() for r in records)


def is_log_record(
    records: list[logging.LogRecord], log_name: str, n_times: int = 0
) -> bool:
    return len([r for r in records if log_name in r.name]) > n_times
