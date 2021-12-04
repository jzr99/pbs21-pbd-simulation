import taichi as ti

@ti.func
def copy(src, dst):
    for I in ti.grouped(src):
        dst[I] = src[I]
