from itertools import chain
import pytest

from paibox.backend.runtime.runtime import *

pexpect = pytest.importorskip("pexpect")


def test_data_decode():
    decoder = ChipOutputDecoder()
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

    output = ChipOutputDecoder.decode_spike_fast(
        shuffle_frame, frame_info, axon_num=axon_num, time_step=timestep
    )

    print("data_decode\n", output)
    print("data_gold  \n", data_inst)

    output = output.flatten()
    data_inst = data_inst.flatten()
    for x, y in zip(output, data_inst):
        if x != 0:
            assert x == y
