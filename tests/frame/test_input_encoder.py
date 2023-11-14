from paibox.frame.input_encoder import DataEncoder
import numpy as np
import time

from paibox.libpaicore.v2.coordinate import Coord
def test_data_encoder():
    encoder = DataEncoder()
    encoder.chip_coord = Coord(0,0)
    encoder.time_step = 1
    encoder.data = np.array([1,2,3,4])
    encoder.frameinfo = np.array([1,2,3,4])
    
    print(encoder.encode())
    
# def test_gen_spike_frame_info():
    
def test_data_encoder_time():
    encoder = DataEncoder()
    encoder.chip_coord = Coord(0,0)
    encoder.time_step = 1
    encoder.data  = np.random.randint(0, 2**64, 1000000, dtype=np.uint64)
    encoder.frameinfo = np.random.randint(0, 2**64, 1000000, dtype=np.uint64)
    
    start = time.time()
    data_frames = encoder.encode()
    end = time.time()
    print("time:",end-start)
