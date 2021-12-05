# test
from render.render import Render
import taichi as ti
from lib.mesh import *
from lib.module import *
from lib.simulator import Simulation


if __name__ == '__main__':
    ti.init()

    # X, Y, Z 对应关系
    # X，Z：平面坐标，X控制水平方向，从左到右依次递增；Z控制竖直方向，从后到前依次递增。（原点在初始视角平面的左上角）
    # Y：纵向坐标
    # rescale：放大缩小使用；translation：尺度变换后，物体的平移
    mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0, 0.6, 0])
    mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0, 1.0, 0])
    mesh_cloth.set_gravity_affected(True)

    module = Module()
    module.add_static_objects(mesh_sphere)
    module.add_simulated_objects(mesh_cloth)

    render = Render({'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()})
    sim = Simulation(module, render)

    sim.run()