from pydantic import ValidationError
import pytest
import random

import numpy as np

from paibox.backend.runtime.libframe import FrameFactory
from paibox.backend.runtime.libframe.frames import *
from paibox.libpaicore import Coord, ReplicationId as RId, FrameHeader as FH


class TestOfflineConfigFrame1:
    @pytest.mark.parametrize(
        "random_seed",
        [
            np.uint64(123456789),
            123456789,
            np.array([123456789], np.uint64),
            np.uint8(123),
            np.uint32(1234567),
        ],
    )
    def test_instance(self, random_seed):
        cf = OfflineConfigFrame1(Coord(1, 0), Coord(3, 4), RId(3, 3), random_seed)

        assert cf.header == FH.CONFIG_TYPE1
        assert cf.random_seed == random_seed
        
        # TODO decode here

    @pytest.mark.parametrize(
        "random_seed",
        [np.array([12345, 6789], np.uint64)],
    )
    def test_instance_illegal(self, random_seed):
        with pytest.raises(TypeError):
            # Must call `int()`
            cf = OfflineConfigFrame1(Coord(1, 0), Coord(3, 4), RId(3, 3), random_seed)


class TestOfflineConfigFrame2:
    def test_instance(self, gen_random_params_reg_dict):
        params_reg_dict = gen_random_params_reg_dict
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        cf = OfflineConfigFrame2(chip_coord, core_coord, rid, params_reg_dict)

        assert cf.header == FH.CONFIG_TYPE2
        assert cf.chip_coord == chip_coord
        assert cf.core_coord == core_coord
        assert cf.rid == rid

        # TODO decode here
        decoded = FrameFactory.decode(cf.value)
        
        assert decoded.header == FH.CONFIG_TYPE2
        assert decoded.chip_coord == chip_coord
        assert decoded.core_coord == core_coord
        assert decoded.rid == rid


    def test_instance_illegal(self, gen_random_params_reg_dict):
        params_reg_dict = gen_random_params_reg_dict
        
        # 1. missing keys
        params_reg_dict.pop("weight_width")
        
        chip_coord, core_coord, rid = Coord(0, 0), Coord(1, 5), RId(2, 2)
        
        with pytest.raises(ValidationError):
            cf = OfflineConfigFrame2(chip_coord, core_coord, rid, params_reg_dict)
            
        # 2. type of value is wrong
        params_reg_dict = gen_random_params_reg_dict
        params_reg_dict["snn_en"] = True
        with pytest.raises(ValidationError):
            cf = OfflineConfigFrame2(chip_coord, core_coord, rid, params_reg_dict)

        # 3. 


    def test_config_frame3(
        self, gen_random_neuron_attr_dict, gen_random_dest_info_dict
    ):
        neuron_attr_dict = gen_random_neuron_attr_dict
        dest_info_dict = gen_random_dest_info_dict

        # chip_coord, core_coord, rid = Coord(0, 1), Coord(1, 1), RId(0, 0)
        # cf = OfflineConfigFrame3(chip_coord, core_coord, rid, params_reg_dict)

        # assert cf.header == FH.CONFIG_TYPE2
        # assert cf.chip_coord == chip_coord
        # assert cf.core_coord == core_coord
        # assert cf.rid == rid
