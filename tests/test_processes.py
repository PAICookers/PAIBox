import pytest

import paibox as pb

def test_Uniform():
    proc = pb.UniformGen(shape_out=(100,))
    
    proc.run(10)