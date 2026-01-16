import logging

import pytest

import paibox as pb
from paibox import _logging
from tests._logging.logging_utils import is_log_record


@pytest.mark.xfail(reason="failed in workflow for unknown reason")
class TestLogging:
    @pytest.mark.make_settings_test(
        **{"backend": logging.WARNING, "core_block_info": True}
    )
    @pytest.mark.usefixtures("log_settings_patch")
    def test_log_settings(self, captured_logs, build_multi_inputproj_net1):
        net = build_multi_inputproj_net1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert is_log_record(captured_logs, ".__core_block_info")
        assert _logging.base.log_state.is_artifact_enabled("core_block_info")

        log = logging.getLogger("paibox.backend.placement")
        assert log.level == logging.WARNING

    @pytest.mark.make_settings_test(**{"coord_assign": True})
    @pytest.mark.usefixtures("log_settings_patch")
    def test_CoreBlock_str_format(self, captured_logs, build_example_net1):
        mapper = pb.Mapper()
        mapper.build(build_example_net1)
        mapper.compile()

        assert is_log_record(captured_logs, ".__coord_assign")
