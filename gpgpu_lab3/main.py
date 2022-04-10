import math
import sys

from numba import cuda

from PIL import Image
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tx", help="threads block size (X)", type=int, default=16)
parser.add_argument("--ty", help="threads block size (Y)", type=int, default=8)
parser.add_argument("--gsize", help="Gauss kernel size", type=int, default=7)
parser.add_argument("--gsigma", help="Gauss kernel sigma", type=int, default=100)
parser.add_argument("input", help="input image")
parser.add_argument("output", help="output image")
args = parser.parse_args()
print(args)
try:
    input_img = Image.open(args.input)
except IOError:
    print("Cannot open input image:", args.input)
    sys.exit(1)

im_array = np.array(input_img)
im_array = np.ascontiguousarray(im_array.transpose(1, 0, 2))

ax = np.linspace(-(args.gsize - 1) / 2., (args.gsize - 1) / 2., args.gsize)
gauss = np.exp(-np.square(ax) / (2 * np.square(args.gsigma)))
kernel = np.outer(gauss, gauss)
kernel = kernel / np.sum(kernel)


@cuda.jit
def gaussian_blur(inp, out, kern):
    x, y = cuda.grid(2)
    if x < inp.shape[0] and y < inp.shape[1]:
        msize = kern.shape[0] // 2
        r = g = b = 0
        for i in range(-msize, msize + 1):
            for j in range(-msize, msize + 1):
                u = x + i
                v = y + j
                if not (0 <= u < inp.shape[0]) or not (0 <= v < inp.shape[1]):
                    u, v = x, y
                r += inp[u, v, 0] * kern[i + msize, j + msize]
                g += inp[u, v, 1] * kern[i + msize, j + msize]
                b += inp[u, v, 2] * kern[i + msize, j + msize]
        out[x, y, 0] = r
        out[x, y, 1] = g
        out[x, y, 2] = b


cuda_input = cuda.to_device(im_array)
cuda_output = cuda.device_array(im_array.shape, dtype=np.uint8)

size_x = math.ceil(im_array.shape[0] / args.tx)
size_y = math.ceil(im_array.shape[1] / args.ty)

from datetime import datetime

start = datetime.now()
gaussian_blur[(size_x, size_y, 1), (args.tx, args.ty, 3)](cuda_input, cuda_output, cuda.to_device(kernel))

cuda_result = cuda_output.copy_to_host()
end = datetime.now()
print("Time:", end - start)

output_img = Image.fromarray(cuda_result.transpose((1, 0, 2)))
output_img.save(args.output)
