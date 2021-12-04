# test
from render.render import Render
import taichi as ti
from lib.mesh import *
from lib.module import *


if __name__ == '__main__':
    ti.init()

    # X, Y, Z 对应关系
    # X，Z：平面坐标，X控制水平方向，从左到右依次递增；Z控制竖直方向，从后到前依次递增。（原点在初始视角平面的左上角）
    # Y：纵向坐标
    # rescale：放大缩小使用；translation：尺度变换后，物体的平移
    mesh_sphere = Mesh(filename='../obj/sphere.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0, 0.6, 0])
    mesh_cloth = Mesh(filename='../obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0, 1.0, 0])

    module = Module()
    module.__add_static_objects(mesh_sphere)
    module.__add_simulated_objects(mesh_cloth)
    sim = Simulation(module)

    render = Render({'sphere': mesh_sphere.export_for_render(), 'cloth': mesh_cloth.export_for_render()})

    while True:
        sim.update()