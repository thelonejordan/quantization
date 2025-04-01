import os
import numpy as np
from tinygrad import Tensor, dtypes

if (seed:=int(os.getenv("SEED", -1))) >= 0: Tensor.manual_seed(seed)

VERBOSE = int(os.getenv("ERROR", 0)) == 0

def quantize_to_int8(a: Tensor):
    # find the normalization constant
    amax = a.abs().max()

    # scale into target range [-127, 127] and round to the nearest
    scaled = a / amax * 127
    int8vals = scaled.round().cast(dtypes.int8)

    # dequantization by rescaling
    fp16vals = int8vals.cast(dtypes.float16) * amax / 127.0

    return int8vals.numpy(), fp16vals.numpy()

def main():
    # vector to quantize
    a = Tensor.randn(8, dtype=dtypes.float16)

    # quantize to int8
    quantized, dequantized = quantize_to_int8(a)

    # information loss
    unquantized = a.numpy()
    quantization_error = np.abs(dequantized - unquantized).mean()
    quantization_std = np.abs(dequantized - unquantized).std()

    if VERBOSE:
        print(f"vec = {unquantized}")
        print("\nquantizing vec...\n")
        for i in range(quantized.size):
            print(f"vec[{i}] = {quantized[i]:>4}  |  {unquantized[i]:>7.4f}  becomes  {dequantized[i]:>7.4f}")
        print()
    print(f"err mean: {quantization_error:>8.6f}")
    print(f"err std: {quantization_std:>9.6f}")
    
    if VERBOSE: print()


if __name__ == "__main__":
    niter = int(os.getenv("COUNT", 1))
    for i in range(niter):
        if niter!=1: print(f"# {i}." + ("\n" if VERBOSE else ""))
        main()
