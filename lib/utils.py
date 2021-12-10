import taichi as ti

@ti.pyfunc
def copy(src, dst):
    for I in ti.ndrange(*src.shape):
        dst[I] = src[I]
