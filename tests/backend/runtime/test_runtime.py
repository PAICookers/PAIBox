import time

import numpy as np
import pytest

pexpect = pytest.importorskip("pexpect")

import paibox as pb
from paibox.backend.runtime.libframe.utils import print_frame
from paibox.backend.runtime.runtime import RuntimeDecoder, RuntimeEncoder
from paibox.libpaicore import Coord, FrameFormat


@pytest.fixture
def input_proj_info():
    return {
        "inp1_1": {
            "addr_core_x": 0,
            "addr_core_y": 0,
            "addr_core_x_ex": 1,
            "addr_core_y_ex": 3,
            "addr_chip_x": 0,
            "addr_chip_y": 0,
            "dest_coords": [
                [
                    Coord(0, 0),
                    Coord(0, 1),
                    Coord(1, 0),
                    Coord(1, 1),
                    Coord(0, 2),
                    Coord(0, 3),
                    Coord(1, 2),
                    Coord(1, 3),
                ]
            ],
        }
    }


def test_data_encoder():
    encoder = RuntimeEncoder()
    encoder.chip_coord = Coord(0, 0)
    encoder.time_step = 1
    encoder.data = np.array([1, 2, 3, 4])
    encoder.frameinfo = np.array([1, 2, 3, 4])

    print(encoder.encode())


# def test_gen_spike_frame_info():


def test_data_encoder_time():
    encoder = RuntimeEncoder()
    encoder.chip_coord = Coord(0, 0)
    encoder.time_step = 1
    encoder.data = np.random.randint(0, 2**64, 1000000, dtype=np.uint64)
    encoder.frameinfo = np.random.randint(0, 2**64, 1000000, dtype=np.uint64)

    start = time.time()
    data_frames = encoder.encode()
    end = time.time()
    print("time:", end - start)


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


def test_data_decode():
    decoder = RuntimeDecoder()
    # frames = np.array([
    #     0b00000_00001_00000_00000_000_000000_00001_00000001_00000001,
    #     0b00000_00001_00000_00000_000_000000_00001_00000111_00000011,
    #     0b00000_00001_00000_00000_000_000000_00001_00000100_00000010,

    #     0b00000_00001_00000_00000_000_000000_00011_00000011_00000010,

    #     0b00000_00001_00000_00000_000_000000_00010_00000001_00000100,
    # ])

    frames = np.array(
        [
            0b00000_00001_00000_00000_000_000000_00001_00000001_00000001,  # 1 1 1 1
            0b00000_00001_00000_00000_000_000000_00001_00000111_00000011,  # 1 1 7 3
            0b00000_00001_00000_00000_000_000000_00001_00000100_00000010,  # 1 1 4 2
            0b00000_00001_00000_00000_000_000000_00011_00000011_00000010,  # 1 3 3 2
            0b00000_00010_00000_00000_000_000000_00010_00000001_00000100,  # 2 2 1 4
        ]
    )
    x = decoder.decode(frames)
    print(x)


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
