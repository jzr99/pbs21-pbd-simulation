import taichi as ti

@ti.func
def copy(src: ti.template, dst: ti.template):
    for I in ti.grouped(src):
        dst[I] = src[I]
