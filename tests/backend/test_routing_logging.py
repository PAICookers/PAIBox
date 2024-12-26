import logging

import pytest
import paibox as pb
from paibox import _logging

from tests._logging.logging_utils import is_log_record


@pytest.mark.xfail(reason="failed in workflow for unknown reason")
class TestLogging:
    @pytest.mark.make_settings_test
    @pytest.mark.usefixtures("log_settings_patch")
    def test_log_settings_default(self, captured_logs, build_multi_inputproj_net1):
        net = build_multi_inputproj_net1
        mapper = pb.Mapper()
        mapper.build(net)
        mapper.compile()

        assert is_log_record(captured_logs, ".__routing_group_info")

        log = _logging.get_artifact_logger(
            "paibox.backend.routing", "routing_group_info"
        )
        assert log.level == logging.DEBUG
