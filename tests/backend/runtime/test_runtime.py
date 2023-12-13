import json
import time
import pytest

import numpy as np
import paibox as pb

from pathlib import Path

from paibox.backend.runtime.libframe.utils import print_frame
from paibox.backend.runtime.runtime import RuntimeDecoder, RuntimeEncoder
from paibox.libpaicore import (
    Coord,
    FrameHeader as FH,
    SpikeFrameFormat as SFF,
    ReplicationId as RId,
)


class TestRuntimeEncoder:
    def test_gen_input_frames_info_by_dict(self):
        fp = Path(__file__).parent / "data"

        with open(fp / "input_proj_info1.json", "r") as f:
            input_proj_info = json.load(f)

        n_input_node = len(input_proj_info.keys())
        assert n_input_node == 2

        common_part = RuntimeEncoder.gen_input_frames_info(
            1, input_proj_info=input_proj_info
        )
        assert len(common_part) == 2

    def test_gen_input_frames_info_by_kwds(self):
        common_part = RuntimeEncoder.gen_input_frames_info(
            (0, 0), 33, RId(0, 0), [0] * 8 + [1] * 8, list(range(16))  # type: ignore
        )

        assert len(common_part) == 16

    def test_encode(self):
        data = list(range(8))
        common_part = RuntimeEncoder.gen_input_frames_info(
            1, Coord(0, 0), Coord(1, 0), RId(0, 0), [0] * 4 + [1] * 4, list(range(8))
        )

        input_spike = RuntimeEncoder.encode(data, common_part)

        data_in_spike = (input_spike >> SFF.DATA_OFFSET) & SFF.DATA_MASK
        # Encode data with none-zero values.
        assert 0 not in data_in_spike

        axons_in_spike = (input_spike >> SFF.AXON_OFFSET) & SFF.AXON_MASK
        # Except axon with data=0
        assert np.array_equal(axons_in_spike, [1, 2, 3, 4, 5, 6, 7])


class TestRuntimeDecoder:
    def test_decode_spike_by_dict(self):
        # oframe_info is `List[FrameArrayType]`, return `List[NDArray[np.uint8]]`
        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
            "n3_1": {
                "5": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 1,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
            # Not occured
            "n4_1": {
                "6": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 2,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
        }
        n_ts = 2
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            n_ts, output_dest_info=output_dest_info
        )

        output_frames = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00001_00000_00000_00000_000_00000000001_00000000_00000010,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
                0b1000_00001_00000_00001_00000_00000_00000_000_00000000100_00000000_00001001,
                0b1000_00001_00000_00001_00000_00000_00000_000_00000000111_00000000_00001010,
            ],
            dtype=np.uint64,
        )
        data = RuntimeDecoder.decode_spike_less1152(
            n_ts, output_frames, oframe_info, flatten=False
        )

        expected = [
            np.array(
                [[0, 0, 0, 7, 0, 8, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
            ),
            np.array(
                [[0, 2, 0, 0, 9, 0, 0, 10], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
            ),
            np.zeros((2, 8), dtype=np.uint8),
        ]
        assert np.array_equal(data, expected)

    def test_decode_spike_by_dict1(self):
        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": [0, 1, 2, 3, 4, 5, 6, 7],
                    "tick_relative": [0] * 8,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            }
        }

        n_ts = 2
        # oframe_info = RuntimeDecoder.gen_output_frames_info(
        #     n_ts, output_dest_info=output_dest_info
        # )
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            n_ts, Coord(1, 0), Coord(0, 0), RId(0, 0), list(range(8))
        )

        output_frames = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
            ],
            dtype=np.uint64,
        )
        data = RuntimeDecoder.decode_spike_less1152(
            n_ts, output_frames, oframe_info, flatten=False
        )

        expected = np.array(
            [[0, 0, 0, 7, 0, 8, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        
        assert np.array_equal(data, expected)

    def test_decode_spike_by_kwds(self):
        # oframe_info is `FrameArrayType`, return `NDArray[np.uint8]`
        # Timestep = 2
        n_axon_max = 100
        n_ts_max = 64
        
        n_ts = 2
        n_axon = 8
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            n_ts, Coord(1, 0), Coord(0, 0), RId(0, 0), list(range(n_axon))
        )

        n_chosen = 3#np.random.randint(1, n_axon * n_ts + 1)

        # choice_idx = np.random.choice(
        #     range(n_axon * n_ts), n_chosen, replace=False
        # )

        choice_idx = [1, 0, 2]
        random = np.random.randint(0, 256, (n_axon * n_ts,), dtype=np.uint8)

        output_frames = oframe_info + random
        shuffle_frame = output_frames[choice_idx]
        shuffle_frame = np.array(
            [
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000000_00000001_00000001,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000011_00000000_00000111,
                0b1000_00001_00000_00000_00000_00000_00000_000_00000000101_00000000_00001000,
            ],
            dtype=np.uint64,
        )
        print_frame(shuffle_frame)

        expected = np.zeros((n_axon * n_ts,), dtype=np.uint8)
        expected[choice_idx] = random[choice_idx]
        expected = np.array(
            [[0, 0, 0, 7, 0, 8, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        data = RuntimeDecoder.decode_spike_less1152(
            n_ts, shuffle_frame, oframe_info, flatten=True
        )

        assert np.array_equal(data, expected)

        # for n_axon in range(1, n_axon_max):
        #     for n_ts in range(1, n_ts_max):
        #         oframe_info = RuntimeDecoder.gen_output_frames_info(
        #             n_ts, Coord(1, 0), Coord(0, 0), RId(0, 0), list(range(n_axon))
        #         )

        #         n_chosen = np.random.randint(1, n_axon * n_ts + 1)

        #         choice_idx = np.random.choice(
        #             range(n_axon * n_ts), n_chosen, replace=False
        #         )

        #         # choice_idx = [1, 0, 2]
        #         random = np.random.randint(0, 256, (n_axon * n_ts,), dtype=np.uint8)

        #         output_frames = oframe_info + random
        #         shuffle_frame = output_frames[choice_idx]

        #         expected = np.zeros((n_axon * n_ts,), dtype=np.uint8)
        #         expected[choice_idx] = random[choice_idx]

        #         data = RuntimeDecoder.decode_spike_less1152(
        #             n_ts, shuffle_frame, oframe_info, flatten=True
        #         )

        #         assert np.array_equal(data, expected)

    def test_decode_spike_perf(self):
        n_axons = 1152

        output_dest_info = {
            "n2_1": {
                "4": {
                    "addr_axon": list(range(n_axons)),
                    "tick_relative": [0] * n_axons,
                    "addr_core_x": 0,
                    "addr_core_y": 0,
                    "addr_core_x_ex": 0,
                    "addr_core_y_ex": 0,
                    "addr_chip_x": 1,
                    "addr_chip_y": 0,
                }
            },
        }
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            1, output_dest_info=output_dest_info
        )
        test_frames = np.zeros((n_axons,), dtype=np.uint64)

        for i in range(n_axons):
            _data = np.random.randint(0, 256, dtype=np.uint8)
            test_frames[i] = (
                (FH.WORK_TYPE1 << SFF.GENERAL_HEADER_OFFSET)
                | (Coord(1, 0).address << SFF.GENERAL_CHIP_ADDR_OFFSET)
                | (i << SFF.AXON_OFFSET)
                | _data
            )

        t1 = time.perf_counter()
        data = RuntimeDecoder.decode_spike_less1152(test_frames, oframe_info)
        t2 = time.perf_counter()

        print(t2 - t1)

    def test_gen_output_frames_info_by_dict1(self):
        fp = Path(__file__).parent / "data"

        with open(fp / "output_dest_info1.json", "r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 1

        common_part = RuntimeDecoder.gen_output_frames_info(
            1, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 800

    def test_gen_output_frames_info_by_dict2(self):
        fp = Path(__file__).parent / "data"

        with open(fp / "output_dest_info2.json", "r") as f:
            output_proj_info = json.load(f)

        n_output_node = len(output_proj_info.keys())
        assert n_output_node == 2

        common_part = RuntimeDecoder.gen_output_frames_info(
            1, output_dest_info=output_proj_info
        )
        assert sum(part.size for part in common_part) == 104

    def test_gen_output_frames_info_by_kwds(self):
        # Overload type.
        oframe_info = RuntimeDecoder.gen_output_frames_info(
            1, (1, 0), (0, 0), (0, 0), [0, 1, 2, 3, 4, 5, 6, 7]
        )

        assert oframe_info.size == 8


# def test_data_encoder_minist():
#     # def parse_mnist(minst_file_addr: str):

#     #     if minst_file_addr is not None:
#     #         minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
#     #         with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
#     #             mnist_file_content = minst_file.read()
#     #         if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
#     #             data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
#     #         else:  # 传入的为图片二进制编码文件地址
#     #             data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8, offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
#     #             data = data.reshape(-1, 28, 28)

#     #     return data

#     train_dataset = torchvision.datasets.MNIST(
#         root="E:/PAIBoxProject/dataset",
#         train=True,
#         transform=torchvision.transforms.ToTensor(),
#         download=False,
#     )
#     img, label = train_dataset[0]
#     # print(img)
#     # img = np.array(img)
#     # plt.title(label)
#     # plt.axis('off')
#     # plt.imshow(img.squeeze(),cmap='gray')
#     # plt.show()

#     start = time.perf_counter()
#     pe = pb.simulator.encoder.PoissonEncoder(shape_out=(28, 28))
#     out = pe.run(10, x=img)
#     # figure = plt.figure()
#     # for i in range(10):
#     #     figure.add_subplot(2,5,i+1)
#     #     plt.title(str(i))
#     #     plt.axis('off')
#     #     plt.imshow(out[i],cmap='gray')
#     # plt.show()

#     de = RuntimeEncoder()

#     frameinfo = (
#         np.random.randint(0, 2**34, 28 * 28 * 10, dtype=np.uint64)
#         << FrameFormat.GENERAL_FRAME_PRE_OFFSET
#     ) & FrameFormat.GENERAL_FRAME_PRE_MASK
#     # print_frame(frameinfo)

#     temp = out.reshape(-1).astype(np.uint64)

#     # print(len(temp))
#     # print(sum(temp))

#     data_frames = de(
#         data=temp, frameinfo=frameinfo, time_step=10, chip_coord=Coord(0, 0)
#     )
#     end = time.perf_counter()
#     print("time:", end - start)
#     work1 = data_frames[:-1]


@pytest.mark.parametrize(
    "timestep,axon_num",
    [
        (2, 3),
        (3, 2),
        (2, 2),
        (3, 3),
        (5, 8),
    ],
)
def test_data_output(timestep, axon_num):
    np.random.seed(1)

    axon_inst = [[i] * timestep for i in range(axon_num)]
    axon_inst = list(chain.from_iterable(axon_inst))

    time_slot_inst = [i for i in range(timestep)] * axon_num
    time_slot_inst = time_slot_inst

    data_inst = np.random.randint(
        0, 2**8 - 1, timestep * axon_num, dtype=np.uint64
    ).reshape(timestep, axon_num)

    frame = OfflineWorkFrame1(
        chip_coord=Coord(0, 0),
        core_coord=Coord(0, 0),
        core_ex_coord=ReplicationId(0, 0),
        axon=axon_inst,
        time_slot=time_slot_inst,
        data=data_inst.flatten(),
    )
    print(frame)
    frame_info = frame.value & ((1 << 64) - 1 - 0b11111111)

    choice_num = np.random.randint(1, timestep * axon_num)
    choice_idx = np.random.choice(range(timestep * axon_num), choice_num, replace=False)
    shuffle_frame = frame.value[choice_idx]

    output = RuntimeDecoder.decode_spike_fast(
        shuffle_frame, frame_info, axon_num=axon_num, time_step=timestep
    )

    print("data_decode\n", output)
    print("data_gold  \n", data_inst)

    output = output.flatten()
    data_inst = data_inst.flatten()
    for x, y in zip(output, data_inst):
        if x != 0:
            assert x == y
