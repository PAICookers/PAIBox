import logging

import pytest

import paibox as pb
from paibox import _logging
from tests._logging.logging_utils import is_log_record


class TestLogging:
    @pytest.mark.make_settings_test
    @pytest.mark.usefixtures("log_settings_patch")
    def test_log_settings_default(self, captured_logs, build_multi_inputproj_net1):
        net = build_multi_inputproj_net1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert is_log_record(captured_logs, ".__cb_axon_grouping")
        assert is_log_record(captured_logs, ".__build_core_blocks")

    @pytest.mark.make_settings_test(
        **{
            "paibox": logging.DEBUG,
            "cb_axon_grouping": True,
            "build_core_blocks": False,
        }
    )
    @pytest.mark.usefixtures("log_settings_patch")
    def test_log_settings_2(self, captured_logs, build_multi_inputproj_net1):
        net = build_multi_inputproj_net1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert is_log_record(captured_logs, ".__cb_axon_grouping")
        assert not is_log_record(captured_logs, ".__build_core_blocks")

        # Artifact logger for coord_assign is enabled by default.
        assert _logging.base.log_state.is_artifact_enabled("cb_axon_grouping")

        log = logging.getLogger("paibox")
        assert log.level == logging.DEBUG

    @pytest.mark.make_settings_test(
        **{"backend": logging.WARNING, "coord_assign": True}
    )
    @pytest.mark.usefixtures("log_settings_patch")
    def test_log_settings_patch(self, captured_logs, build_multi_inputproj_net1):
        net = build_multi_inputproj_net1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert is_log_record(captured_logs, ".__coord_assign")
        # Artifact logger for cb_axon_grouping is disabed.
        assert not _logging.base.log_state.is_artifact_enabled("cb_axon_grouping")

        log = logging.getLogger("paibox.backend.mapper")
        assert log.level == logging.WARNING
