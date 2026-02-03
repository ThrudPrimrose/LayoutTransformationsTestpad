import math
import typing

strides = [1, 64]
tile_dims = [8, 8]


def distance(i: int, j: int, strides: typing.Tuple[int, int]):
    d1 = abs(j - i) // tile_dims[0]
    d0 = abs(j - i) % tile_dims[0]

    return d1 * strides[1] + d0

avg_dist = 0
for i in range(tile_dims[0] * tile_dims[1]):
    for j in range(tile_dims[0] * tile_dims[1]):
        d = distance(i, j, strides)
        avg_dist += d
        print(f"Distance between {i} and {j} offset elements is {d}")

avg_dist = avg_dist / ((tile_dims[0] * tile_dims[1])**2)
print("Avg distance", avg_dist)

def distance(i: int, j: int, strides: typing.Tuple[int, int]):
    d1 = abs(j - i) // tile_dims[0]
    d0 = abs(j - i) % tile_dims[0]

    return d1 * strides[1] + d0

strides = [1, ]
tile_dims = [8, ]
def distance1d(i: int, j: int, strides: typing.Tuple[int, int]):
    d1 = abs(j - i) // tile_dims[0]
    d0 = abs(j - i) % tile_dims[0]

    return d1 * 1 + d0


avg_dist = 0
for i in range(tile_dims[0]):
    for j in range(tile_dims[0]):
        d = distance1d(i, j, strides)
        avg_dist += d
        print(f"Distance between {i} and {j} offset elements is {d}")

avg_dist = avg_dist / ((tile_dims[0]*tile_dims[0]))
print("Avg distance", avg_dist)