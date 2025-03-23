from pathlib import Path

import numpy as np
import pytest

import paibox as pb
from paibox.types import NEUOUT_U8_DTYPE, VOLTAGE_DTYPE
from tests.components.utils import ann_bit_trunc, conv1d_golden

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
CONFIG_DIR = TEST_DIR / "config"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

FIXED_RNG = np.random.default_rng(seed=42)


def _out_bypass1(t, data1, *args, **kwargs):
    return data1


class TestOnBoard_WRAMMapping:
    def test_001(self):
        class Net001(pb.Network):
            def __init__(self, w):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape[0],))
                # Start at ts=1, end at ts=1+2=3
                self.l1 = pb.Linear(
                    self.i1, shape[1], w, tick_wait_start=1, tick_wait_end=2
                )
                self.p1 = pb.Probe(self.l1, "feature_map")

        TEST_NAME = self.test_001.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")
        shape = (144, 400)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-10, 10, size=shape, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 5, size=(sim_time, shape[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape[1]), dtype=np.uint8)

        network = Net001(weight1)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            if i < 2:
                # At ts = 1 & 2, there is output data
                ref = ann_bit_trunc(
                    inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE),
                    bit_trunc=network.l1.bit_trunc,
                )
                assert np.array_equal(sim.data[network.p1][i], ref)
            else:
                # At ts > 2, linear is not working, no output data
                ref = np.zeros_like(sim.data[network.p1][i])

            assert np.array_equal(sim.data[network.p1][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p1][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_002(self):
        class Net002(pb.Network):
            def __init__(self, w):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape[1], w, tick_wait_start=1, tick_wait_end=0
                )
                self.p1 = pb.Probe(self.l1, "feature_map")

        TEST_NAME = self.test_002.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")
        shape = (500, 100)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-10, 10, size=shape, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 5, size=(sim_time, shape[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape[1]), dtype=np.uint8)

        network = Net002(weight1)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            ref = ann_bit_trunc(
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE),
                bit_trunc=network.l1.bit_trunc,
            )
            assert np.array_equal(sim.data[network.p1][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p1][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_003(self):
        class Net003(pb.Network):
            def __init__(self, w):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape[1], w, bias=99, tick_wait_start=1, tick_wait_end=0
                )
                self.p1 = pb.Probe(self.l1, "feature_map")

        TEST_NAME = self.test_003.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")
        shape = (240, 200)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=2, enable weight bit optimization
            weight1 = FIXED_RNG.integers(-2, 2, size=shape, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 2, size=(sim_time, shape[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape[1]), dtype=np.uint8)

        network = Net003(weight1)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            ref = ann_bit_trunc(
                # Use bias in linear
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE)
                + network.l1.bias,
                bit_trunc=network.l1.bit_trunc,
            )
            assert np.array_equal(sim.data[network.p1][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p1][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=True)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_004(self):
        class Net004(pb.Network):
            def __init__(self, w):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape[1], w, tick_wait_start=1, tick_wait_end=0
                )
                self.p1 = pb.Probe(self.l1, "feature_map")

        TEST_NAME = self.test_004.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")
        shape = (240, 500)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=4, enable weight bit optimization
            weight1 = FIXED_RNG.integers(-8, 8, size=shape, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 5, size=(sim_time, shape[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape[1]), dtype=np.uint8)

        network = Net004(weight1)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            ref = ann_bit_trunc(
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE),
                bit_trunc=network.l1.bit_trunc,
            )
            assert np.array_equal(sim.data[network.p1][i], ref)
            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p1][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=True)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_005(self):
        class Net005(pb.Network):
            def __init__(self, w1, w2):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape1[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape1[1], w1, tick_wait_start=1, tick_wait_end=0
                )
                # Start at ts=2, no end
                self.l2 = pb.Linear(
                    self.l1,
                    shape2[1],
                    w2,
                    bias=2,
                    bit_trunc=9,
                    tick_wait_start=2,
                    tick_wait_end=0,
                )
                self.p1 = pb.Probe(self.l1, "feature_map")
                self.p2 = pb.Probe(self.l2, "feature_map")

        TEST_NAME = self.test_005.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (200, 200)
        shape2 = (shape1[1], 10)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-5, 5, size=shape1, dtype=np.int8)
            # W=4
            weight2 = FIXED_RNG.integers(-15, 15, size=shape2, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 5, size=(sim_time, shape1[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape2[1]), dtype=np.uint8)

        network = Net005(weight1, weight2)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            # Use bias in linear1
            _l1 = ann_bit_trunc(
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE)
                + network.l1.bias,
                bit_trunc=network.l1.bit_trunc,
            )
            # The miintermidiate result is correct
            assert np.array_equal(sim.data[network.p1][i], _l1)

            if i > 0:
                # The input of Linear2 is the output of Linear1 at the last timestamp
                ref = ann_bit_trunc(
                    sim.data[network.p1][i - 1] @ weight2.astype(VOLTAGE_DTYPE)
                    + network.l2.bias,
                    bit_trunc=network.l2.bit_trunc,
                )
                # At ts >= 2, Linear2 is outputing
                assert np.array_equal(sim.data[network.p2][i], ref)
            else:
                # At ts = 1, Linear2 is not working, no output data
                ref = np.zeros_like(sim.data[network.p2][i])

            assert np.array_equal(sim.data[network.p2][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p2][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_006(self):
        class Net006(pb.Network):
            def __init__(self, w1, w2):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape1[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape1[1], w1, tick_wait_start=1, tick_wait_end=0
                )
                # Start at ts=2, no end
                self.l2 = pb.Linear(
                    self.l1, shape2[1], w2, tick_wait_start=2, tick_wait_end=0
                )
                self.p1 = pb.Probe(self.l1, "feature_map")
                self.p2 = pb.Probe(self.l2, "feature_map")

        TEST_NAME = self.test_006.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (240, 100)
        shape2 = (shape1[1], 10)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=1, enable weight bit optimization
            weight1 = FIXED_RNG.integers(
                0, 1, size=shape1, dtype=np.int8, endpoint=True
            )
            # W=1
            weight2 = FIXED_RNG.integers(
                0, 1, size=shape2, dtype=np.int8, endpoint=True
            )
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 5, size=(sim_time, shape1[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape2[1]), dtype=np.uint8)

        network = Net006(weight1, weight2)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            _l1 = ann_bit_trunc(
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE),
                bit_trunc=network.l1.bit_trunc,
            )
            # The miintermidiate result is correct
            assert np.array_equal(sim.data[network.p1][i], _l1)

            if i > 0:
                # The input of Linear2 is the output of Linear1 at the last timestamp
                ref = ann_bit_trunc(
                    sim.data[network.p1][i - 1] @ weight2.astype(VOLTAGE_DTYPE),
                    bit_trunc=network.l2.bit_trunc,
                )
                # At ts >= 2, Linear2 is outputing
                assert np.array_equal(sim.data[network.p2][i], ref)
            else:
                # At ts = 1, Linear2 is not working, no output data
                ref = np.zeros_like(sim.data[network.p2][i])

            assert np.array_equal(sim.data[network.p2][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p2][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=True)
        mapper.export(fp=CONFIG_CASE_DIR, format="bin", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_007(self):
        class Net007(pb.Network):
            def __init__(self, w1):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape1[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape1[1], w1, tick_wait_start=1, tick_wait_end=0
                )
                self.p1 = pb.Probe(self.l1, "feature_map")

        TEST_NAME = self.test_007.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (120, 800)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=4, enable weight bit optimization
            weight1 = FIXED_RNG.integers(-8, 8, size=shape1, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 3, size=(sim_time, shape1[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape1[1]), dtype=np.uint8)

        network = Net007(weight1)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            ref = ann_bit_trunc(
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE),
                bit_trunc=network.l1.bit_trunc,
            )

            assert np.array_equal(sim.data[network.p1][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p1][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=True)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_008(self):
        class Net008(pb.Network):
            def __init__(self, w1, w2):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(shape1[0],))
                # Start at ts=1, no end
                self.l1 = pb.Linear(
                    self.i1, shape1[1], w1, tick_wait_start=1, tick_wait_end=0
                )
                # Start at ts=2, no end
                self.l2 = pb.Linear(
                    self.l1, shape2[1], w2, tick_wait_start=2, tick_wait_end=0
                )
                self.p1 = pb.Probe(self.l1, "feature_map")
                self.p2 = pb.Probe(self.l2, "feature_map")

        TEST_NAME = self.test_008.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (120, 600)
        shape2 = (shape1[1], 10)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=4, enable weight bit optimization
            weight1 = FIXED_RNG.integers(-8, 8, size=shape1, dtype=np.int8)
            # W=8
            weight2 = FIXED_RNG.integers(-15, 15, size=shape2, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                np.iinfo(np.uint8).min, 5, size=(sim_time, shape1[0]), dtype=np.uint8
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, shape2[1]), dtype=np.uint8)

        network = Net008(weight1, weight2)
        sim = pb.Simulator(network, start_time_zero=False)

        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

        # Check
        for i in range(sim_time):
            _l1 = ann_bit_trunc(
                inpdata1[i, :].ravel() @ weight1.astype(VOLTAGE_DTYPE),
                bit_trunc=network.l1.bit_trunc,
            )
            # The miintermidiate result is correct
            assert np.array_equal(sim.data[network.p1][i], _l1)

            if i > 0:
                # The input of Linear2 is the output of Linear1 at the last timestamp
                ref = ann_bit_trunc(
                    sim.data[network.p1][i - 1] @ weight2.astype(VOLTAGE_DTYPE),
                    bit_trunc=network.l2.bit_trunc,
                )
                # At ts >= 2, Linear2 is outputing
                assert np.array_equal(sim.data[network.p2][i], ref)
            else:
                # At ts = 1, Linear2 is not working, no output data
                ref = np.zeros_like(sim.data[network.p2][i])

            assert np.array_equal(sim.data[network.p2][i], ref)

            if USE_EXISTING_DATA:
                assert np.array_equal(ref, refresult1[i, :])
            else:
                refresult1[i, :] = sim.data[network.p2][i]

            print(f"t={i + 1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=True)
        mapper.export(fp=CONFIG_CASE_DIR, format="bin", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")


class TestOnBoard_SpikingOp:
    def test_001_Conv1d(self):
        class Net001(pb.Network):
            def __init__(self, w1):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=shape1)
                # Start at ts=1, no end
                self.n1 = pb.IF(out_shape, 10, tick_wait_start=1, tick_wait_end=0)
                self.conv = pb.Conv1d(self.i1, self.n1, w1)
                self.p1 = pb.Probe(self.n1, "feature_map")

        TEST_NAME = self.test_001_Conv1d.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (8, 100)  # C*L
        ksize = (4, shape1[0], 8)  # O*C*K
        out_shape = (4, 93)

        sim_time = 5

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-10, 12, size=ksize, dtype=np.int8)
            inpdata1 = FIXED_RNG.integers(
                0, 1, size=(sim_time,) + shape1, dtype=np.bool_, endpoint=True
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time,) + out_shape, dtype=np.bool_)

        network = Net001(weight1)
        sim = pb.Simulator(network, start_time_zero=False)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[i, :])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[network.p1][i]

            print(f"t={i + 1}\n", sim.data[network.p1][i])

        # Check
        # TODO the result of conv1d is supposed to pass to LIF
        # for i in range(sim_time):
        #     ref = conv1d_golden(inpdata1[i, :], (out_shape[1],), weight1, (1,), (0,))
        #     assert np.array_equal(sim.data[network.p1][i], ref)

        #     if USE_EXISTING_DATA:
        #         assert np.array_equal(ref, refresult1[i, :])
        #     else:
        #         refresult1[i, :] = sim.data[network.p2][i]

        #     print(f"t={i+1}\n", ref)

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")


class TestOnBoard_SemiFoldedOp:
    def test_001_Conv2dSemiFolded(self):
        class Net001(pb.DynSysGroup):
            def __init__(self, w1):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 1, 0, tick_wait_start=1)

        TEST_NAME = self.test_001_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (1, 64, 64)  # C*H*W
        ksize = (4, shape1[0], 7, 7)  # O*C*K*K
        out_shape = (4, 58, 58)

        sim_time = 65

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-10, 10, size=ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net001(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    # 对比test002-005系列
    # weight正常
    def test_002_Conv2dSemiFolded(self):
        class Net002(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 2, 0, tick_wait_start=1)

        TEST_NAME = self.test_002_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (1, 8, 8)  # C*H*W
        ksize = (1, 1, 2, 2)  # O*C*K*k
        out_shape = (1, 4, 4)

        sim_time = 16

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("input:", inpdata1)
            print("weight:", weight1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            weight1 = FIXED_RNG.integers(-10, 10, size=ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net002(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    # weight全为1
    def test_003_Conv2dSemiFolded(self):
        class Net003(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 2, 0, tick_wait_start=1)

        TEST_NAME = self.test_003_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (1, 8, 8)  # C*H*W
        ksize = (1, shape1[0], 2, 2)  # O*C*K*k
        out_shape = (1, 4, 4)

        sim_time = 16

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("Input", inpdata1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # weight1 = FIXED_RNG.integers(0, 1, size=ksize, dtype=np.int8)
            weight1 = np.ones(ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net003(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]
            print(sim.data[probe][i].shape)
            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    # 扇入扩展， weight全正1
    def test_004_Conv2dSemiFolded(self):
        class Net004(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 2, 0, tick_wait_start=1)

        TEST_NAME = self.test_004_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (8, 64, 64)  # C*H*W
        ksize = (4, shape1[0], 7, 7)  # O*C*K*k
        out_shape = (4, 29, 29)

        sim_time = 65

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("Input", inpdata1)
            print("weight", weight1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # weight1 = FIXED_RNG.integers(0, 1, size=ksize, dtype=np.int8)
            weight1 = np.ones(ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net004(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]
            print(sim.data[probe][i].shape)
            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    # 扇入扩展
    def test_005_Conv2dSemiFolded(self):
        class Net005(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 2, 0, tick_wait_start=1)

        TEST_NAME = self.test_005_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (8, 64, 64)  # C*H*W
        ksize = (4, shape1[0], 7, 7)  # O*C*K*k
        out_shape = (4, 29, 29)

        sim_time = 65

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("Input", inpdata1)
            print("weight", weight1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            weight1 = FIXED_RNG.integers(0, 5, size=ksize, dtype=np.int8)
            # weight1 = np.ones(ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net005(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]
            print(sim.data[probe][i].shape)
            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    # 对比006-009
    def test_006_Conv2dSemiFolded(self):
        class Net006(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 1, 1, tick_wait_start=1)

        TEST_NAME = self.test_006_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (1, 8, 8)  # C*H*W
        ksize = (1, shape1[0], 3, 3)  # O*C*K*k
        out_shape = (1, 8, 8)

        sim_time = 10

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("Input:", inpdata1)
            print("weight:", weight1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-10, 10, size=ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net006(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_007_Conv2dSemiFolded(self):
        class Net007(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 1, 1, tick_wait_start=1)

        TEST_NAME = self.test_007_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (1, 8, 8)  # C*H*W
        ksize = (1, shape1[0], 3, 3)  # O*C*K*k
        out_shape = (1, 8, 8)

        sim_time = 10

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("Input:", inpdata1)
            print("weight:", weight1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # weight =1
            # weight1 = FIXED_RNG.integers(-10, 10, size=ksize, dtype=np.int8)
            weight1 = np.ones(ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net007(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_008_Conv2dSemiFolded(self):
        class Net008(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 1, 1, tick_wait_start=1)

        TEST_NAME = self.test_008_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (8, 32, 32)  # C*H*W
        ksize = (4, shape1[0], 3, 3)  # O*C*K*k
        out_shape = (4, 32, 32)

        sim_time = 35

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            print("Input:", inpdata1)
            print("weight:", weight1)
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # weight =1
            # weight1 = FIXED_RNG.integers(-10, 10, size=ksize, dtype=np.int8)
            weight1 = np.ones(ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net008(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_009_Conv2dSemiFolded(self):
        class Net009(pb.DynSysGroup):
            def __init__(self, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w2, 1, 1, tick_wait_start=1)

        TEST_NAME = self.test_009_Conv2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (8, 32, 32)  # C*H*W
        ksize = (4, shape1[0], 3, 3)  # O*C*K*k
        out_shape = (4, 32, 32)

        sim_time = 35

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(-10, 10, size=ksize, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net009(weight1)
        conv2d = network.conv1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[conv2d][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE, weight1=weight1, inpdata1=inpdata1, refresult1=refresult1
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_010_MaxPool2dSemiFolded(self):
        class Net010(pb.DynSysGroup):
            def __init__(self, ksize):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.pool1 = pb.MaxPool2dSemiFolded(
                    self.i1, ksize, 2, tick_wait_start=1
                )

        TEST_NAME = self.test_010_MaxPool2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize = (2, 2)  # O*C*K*K
        out_shape = (3, 16, 16)

        sim_time = 32

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"

        try:
            npz = np.load(NPZ_FILE)
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net010(ksize)
        pool = network.pool1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[pool][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(NPZ_FILE, inpdata1=inpdata1, refresult1=refresult1)

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_011_AvgPool2dSemiFolded(self):
        class Net011(pb.DynSysGroup):
            def __init__(self, ksize):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.pool1 = pb.AvgPool2dSemiFolded(
                    self.i1, ksize, 2, 0, tick_wait_start=1
                )

        TEST_NAME = self.test_011_AvgPool2dSemiFolded.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize = (2, 2)  # O*C*K*K
        out_shape = (3, 16, 16)

        sim_time = 32

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"

        try:
            npz = np.load(NPZ_FILE)
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros(
                (sim_time, out_shape[0] * out_shape[1]), dtype=NEUOUT_U8_DTYPE
            )

        network = Net011(ksize)
        pool = network.pool1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe = pb.Probe(generated[pool][0], "output")
        sim.add_probe(probe)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe][i]

            print(f"t={i + 1}\n", sim.data[probe][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(NPZ_FILE, inpdata1=inpdata1, refresult1=refresult1)

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_012_Conv2dSemiFoldedNet(self):
        class Net012(pb.DynSysGroup):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 1, 1, tick_wait_start=1)

                self.conv2 = pb.Conv2dSemiFolded(
                    self.conv1, w2, 1, 1, tick_wait_start=3
                )

                self.linear1 = pb.LinearSemiFolded(
                    self.conv2, out_shape[1], weights=w3, bias=2, tick_wait_start=5
                )

        TEST_NAME = self.test_012_Conv2dSemiFoldedNet.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize1 = (4, shape1[0], 3, 3)  # O*C*K*K
        ksize2 = (4, ksize1[0], 3, 3)
        out_shape = (4 * 32 * 32, 10)

        sim_time = 40

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"

        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            weight3 = npz["weight3"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(0, 3, size=ksize1, dtype=np.int8)
            weight2 = FIXED_RNG.integers(-3, 3, size=ksize2, dtype=np.int8)
            weight3 = FIXED_RNG.integers(-3, 5, size=out_shape, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, out_shape[1]), dtype=NEUOUT_U8_DTYPE)

        network = Net012(weight1, weight2, weight3)
        conv2d1 = network.conv1
        conv2d2 = network.conv2
        linear = network.linear1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe1 = pb.Probe(generated[conv2d1][0], "output")
        probe2 = pb.Probe(generated[conv2d2][0], "output")
        probe3 = pb.Probe(generated[linear][0], "output")

        sim.add_probe(probe1)
        sim.add_probe(probe2)
        sim.add_probe(probe3)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe3][i]

            print(f"t={i + 1}\n", sim.data[probe3][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                weight3=weight3,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_013_Conv2dSemiFoldedNet(self):
        class Net013(pb.DynSysGroup):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 2, 1, tick_wait_start=1)
                self.conv2 = pb.Conv2dSemiFolded(
                    self.conv1, w2, 2, 1, tick_wait_start=3
                )
                self.linear1 = pb.LinearSemiFolded(
                    self.conv2, out_shape[1], weights=w3, bias=2, tick_wait_start=5
                )

        TEST_NAME = self.test_013_Conv2dSemiFoldedNet.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize1 = (4, shape1[0], 4, 4)  # O*C*K*K
        ksize2 = (4, ksize1[0], 4, 4)
        out_shape = (4 * 8 * 8, 10)

        sim_time = 40

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"
        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            weight3 = npz["weight3"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(0, 3, size=ksize1, dtype=np.int8)
            weight2 = FIXED_RNG.integers(-3, 3, size=ksize2, dtype=np.int8)
            weight3 = FIXED_RNG.integers(-3, 5, size=out_shape, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, out_shape[1]), dtype=NEUOUT_U8_DTYPE)

        network = Net013(weight1, weight2, weight3)
        conv2d1 = network.conv1
        conv2d2 = network.conv2
        linear = network.linear1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe1 = pb.Probe(generated[conv2d1][0], "output")
        probe2 = pb.Probe(generated[conv2d2][0], "output")
        probe3 = pb.Probe(generated[linear][0], "output")

        sim.add_probe(probe1)
        sim.add_probe(probe2)
        sim.add_probe(probe3)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe3][i]

            print(f"t={i + 1}\n", sim.data[probe3][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                weight3=weight3,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_014_CNNSemiFoldedNet(self):
        class Net014(pb.DynSysGroup):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])

                self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 1, 1, tick_wait_start=1)
                self.pool1 = pb.MaxPool2dSemiFolded(
                    self.conv1, (2, 2), 2, tick_wait_start=3
                )
                self.conv2 = pb.Conv2dSemiFolded(
                    self.pool1, w2, 1, 1, tick_wait_start=5
                )
                self.pool2 = pb.MaxPool2dSemiFolded(
                    self.conv2, (2, 2), 2, tick_wait_start=7
                )
                self.linear1 = pb.LinearSemiFolded(
                    self.pool2, out_shape[1], weights=w3, bias=2, tick_wait_start=9
                )

        TEST_NAME = self.test_014_CNNSemiFoldedNet.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize1 = (4, shape1[0], 3, 3)  # O*C*K*K
        ksize2 = (4, ksize1[0], 3, 3)
        out_shape = (4 * 8 * 8, 10)

        sim_time = 42

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"

        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            weight3 = npz["weight3"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(0, 3, size=ksize1, dtype=np.int8)
            weight2 = FIXED_RNG.integers(-3, 3, size=ksize2, dtype=np.int8)
            weight3 = FIXED_RNG.integers(-3, 5, size=out_shape, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, out_shape[1]), dtype=NEUOUT_U8_DTYPE)

        network = Net014(weight1, weight2, weight3)
        conv2d1 = network.conv1
        conv2d2 = network.conv2
        linear = network.linear1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe1 = pb.Probe(generated[conv2d1][0], "output")
        probe2 = pb.Probe(generated[conv2d2][0], "output")
        probe3 = pb.Probe(generated[linear][0], "output")

        sim.add_probe(probe1)
        sim.add_probe(probe2)
        sim.add_probe(probe3)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe3][i]

            print(f"t={i + 1}\n", sim.data[probe3][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                weight3=weight3,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_015_Conv2dSemiFoldedNet(self):
        class Net015(pb.DynSysGroup):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 2, 1, tick_wait_start=1)

                self.conv2 = pb.Conv2dSemiFolded(
                    self.conv1, w2, 2, 1, tick_wait_start=3
                )

                self.linear1 = pb.LinearSemiFolded(
                    self.conv2,
                    out_shape[1],
                    weights=w3,
                    bias=2,
                    tick_wait_start=5,
                    rin_buffer_option=True,
                )

        TEST_NAME = self.test_015_Conv2dSemiFoldedNet.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize1 = (4, shape1[0], 4, 4)  # O*C*K*K
        ksize2 = (4, ksize1[0], 4, 4)
        out_shape = (4 * 8 * 8, 10)

        sim_time = 40

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"

        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            weight3 = npz["weight3"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(0, 3, size=ksize1, dtype=np.int8)
            weight2 = FIXED_RNG.integers(-3, 3, size=ksize2, dtype=np.int8)
            weight3 = FIXED_RNG.integers(-3, 5, size=out_shape, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, out_shape[1]), dtype=NEUOUT_U8_DTYPE)

        network = Net015(weight1, weight2, weight3)
        conv2d1 = network.conv1
        conv2d2 = network.conv2
        linear = network.linear1
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe1 = pb.Probe(generated[conv2d1][0], "output")
        probe2 = pb.Probe(generated[conv2d2][0], "output")
        probe3 = pb.Probe(generated[linear][0], "output")

        sim.add_probe(probe1)
        sim.add_probe(probe2)
        sim.add_probe(probe3)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe3][i]

            print(f"t={i + 1}\n", sim.data[probe3][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                weight3=weight3,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")

    def test_016_Conv2dSemiFoldedNet(self):
        class Net016(pb.DynSysGroup):
            def __init__(self, w1, w2):
                super().__init__()
                self.i1 = pb.InputProj(input=_out_bypass1, shape_out=shape1[:2])
                self.conv1 = pb.Conv2dSemiFolded(self.i1, w1, 1, 1, tick_wait_start=1)

                self.conv2 = pb.Conv2dSemiFolded(
                    self.conv1, w2, 1, 1, tick_wait_start=3
                )

                # self.linear1 = pb.LinearSemiFolded(
                #     self.conv2, out_shape[1], weights=w3, bias=2, tick_wait_start=5
                # )

        TEST_NAME = self.test_016_Conv2dSemiFoldedNet.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        shape1 = (3, 32, 32)  # C*H*W
        ksize1 = (4, shape1[0], 3, 3)  # O*C*K*K
        ksize2 = (4, ksize1[0], 3, 3)
        out_shape = 4 * 32

        sim_time = 40

        USE_EXISTING_DATA = False
        NPZ_FILE = TEST_CASE_DIR / "data.npz"

        try:
            npz = np.load(NPZ_FILE)
            weight1 = npz["weight1"]
            weight2 = npz["weight2"]
            inpdata1 = npz["inpdata1"]
            refresult1 = npz["refresult1"]
            print("Using the existing data file")
            USE_EXISTING_DATA = True
        except:
            pass

        if not USE_EXISTING_DATA:
            print("Generating new data")
            # W=8, disable weight bit optimization
            weight1 = FIXED_RNG.integers(0, 3, size=ksize1, dtype=np.int8)
            weight2 = FIXED_RNG.integers(-3, 3, size=ksize2, dtype=np.int8)
            inpa = FIXED_RNG.integers(0, 4, size=shape1, dtype=NEUOUT_U8_DTYPE)
            inpdata1 = np.concatenate(
                [inpa, np.zeros_like(inpa)], axis=2, dtype=inpa.dtype
            )
            # Shape of reference result is sim_time * refdata
            refresult1 = np.zeros((sim_time, out_shape), dtype=NEUOUT_U8_DTYPE)

        network = Net016(weight1, weight2)
        conv2d1 = network.conv1
        conv2d2 = network.conv2
        generated = network.build_modules()
        sim = pb.Simulator(network, start_time_zero=False)
        probe1 = pb.Probe(generated[conv2d1][0], "output")
        probe2 = pb.Probe(generated[conv2d2][0], "output")

        sim.add_probe(probe1)
        sim.add_probe(probe2)
        for i in range(sim_time):
            pb.FRONTEND_ENV.save(data1=inpdata1[:, :, i])
            sim.run(1)

            if not USE_EXISTING_DATA:
                refresult1[i, :] = sim.data[probe2][i]

            print(f"t={i + 1}\n", sim.data[probe2][i])

        # Save weights & input data
        if not USE_EXISTING_DATA:
            np.savez(
                NPZ_FILE,
                weight1=weight1,
                weight2=weight2,
                inpdata1=inpdata1,
                refresult1=refresult1,
            )

        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(fp=CONFIG_CASE_DIR, format="txt", use_hw_sim=True)

        print(f"Test {TEST_NAME} end")


class TestOnBoard_ReadNeuronVoltage:
    # Test cases for reading neuron voltage. Don't care the weights.
    def test_001_one_onode(self):
        # 1 output node on 4 cores
        class Net001(pb.Network):
            def __init__(self):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(100,))
                self.n1 = pb.IF((200,), 127, tick_wait_start=1)
                self.s1 = pb.FullConn(
                    self.i1,
                    self.n1,
                    FIXED_RNG.integers(-10, 10, size=(100, 200), dtype=np.int8),
                )

        TEST_NAME = self.test_001_one_onode.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        network = Net001()
        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(
            fp=CONFIG_CASE_DIR, format="bin", use_hw_sim=True, read_voltage=network.n1
        )

        print(f"Test {TEST_NAME} end")

    def test_002_one_onode_lcn(self):
        # 1 output node on 4 cores, lcn > 1
        class Net002(pb.Network):
            def __init__(self):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(2000,))
                self.n1 = pb.IF((100,), 127, tick_wait_start=1)
                self.s1 = pb.FullConn(
                    self.i1,
                    self.n1,
                    FIXED_RNG.integers(-10, 10, size=(2000, 100), dtype=np.int8),
                )

        TEST_NAME = self.test_002_one_onode_lcn.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        network = Net002()
        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(
            fp=CONFIG_CASE_DIR, format="bin", use_hw_sim=True, read_voltage=network.n1
        )

        print(f"Test {TEST_NAME} end")

    def test_003_one_onode_lcn_ann(self):
        # 1 output node on 4 cores, lcn > 1, ANN mode
        class Net003(pb.Network):
            def __init__(self):
                super().__init__()
                self.i1 = pb.InputProj(_out_bypass1, shape_out=(400,))
                self.n1 = pb.ANNNeuron((400,), tick_wait_start=1)
                self.s1 = pb.FullConn(
                    self.i1,
                    self.n1,
                    FIXED_RNG.integers(-10, 10, size=(400, 400), dtype=np.int8),
                )

        TEST_NAME = self.test_003_one_onode_lcn_ann.__name__
        TEST_CASE_DIR = DATA_DIR / TEST_NAME
        CONFIG_CASE_DIR = CONFIG_DIR / TEST_NAME
        if not TEST_CASE_DIR.exists():
            TEST_CASE_DIR.mkdir()

        print(f"\nTest {TEST_NAME} start")

        network = Net003()
        mapper = pb.Mapper()
        mapper.build(network)
        mapper.compile(weight_bit_optimization=False)
        mapper.export(
            fp=CONFIG_CASE_DIR, format="bin", use_hw_sim=True, read_voltage=network.n1
        )

        print(f"Test {TEST_NAME} end")


if __name__ == "__main__":
    # NOTE: run test cases by cli
    # For example:
    # >>> cd paibox
    # >>> poetry run python ./tests/on_board/test_onboard.py
    test_fp = Path.cwd() / Path(__file__)
    test_class = "TestOnBoard_WRAMMapping"
    test_case_name = "test_007"
    # Run a specific test case
    retcode = pytest.main(["-s", f"{test_fp}::{test_class}::{test_case_name}"])
