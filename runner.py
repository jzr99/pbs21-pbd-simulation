# test
from render.render import Render
import taichi as ti
from lib.mesh import *
from lib.module import *
from lib.simulator import Simulation


if __name__ == '__main__':
    device = ti.cpu  # todo
    # device = ti.cuda
    ti.init()

    # X, Y, Z 对应关系
    # X，Z：平面坐标，X控制水平方向，从左到右依次递增；Z控制竖直方向，从后到前依次递增。（原点在初始视角平面的左上角）
    # Y：纵向坐标
    # rescale：放大缩小使用；translation：尺度变换后，物体的平移
    # !!!!!!! _large物体由open3d导出，其triangle的索引顺序与小文件的相反，会导致平面法向量计算相反，渲染的颜色面出错
    # mesh_sphere = Mesh(filename='./obj/sphere_large.obj', color=[1.0, 0.4, 0.2], translation=[0, 0.7, 0], reverse_triangle_verts=True)
    # mesh_sphere = Mesh(filename='./obj/sphere_large_new.obj', color=[1.0, 0.4, 0.2], translation=[0, 0.7, 0], reverse_triangle_verts=True)
    # mesh_cloth = Mesh(filename='./obj/cloth_large.obj', color=[0.2, 0.2, 0.2], translation=[0, 1.0, 0], reverse_triangle_verts=True)
    mesh_cloth = Mesh(filename='./obj/cloth_large_new.obj', color=[0.2, 0.2, 0.2], rescale=0.6, translation=[0, 0.7, 0], reverse_triangle_verts=True)
    # mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[1.0, 0.4, 0.2], rescale=0.1, translation=[0, 0.7, 0])
    mesh_bunny = Mesh(filename='./obj/bunny.obj', color=[0.5, 0.7, 0.99], rescale=5, translation=[0, 0.2, 0])
    # mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.2, translation=[0, 1.1, 0])
    mesh_ground = Mesh(filename='./obj/ground.obj', color=[0.9, 0.9, 0.9], rescale=1.0, translation=[0, 0, 0])
    mesh_cloth.set_gravity_affected(True)
    mesh_cloth.set_wind_affected(False)

    module = Module()
    # module.add_static_objects(mesh_sphere)
    module.add_static_objects(mesh_bunny)
    module.add_static_objects(mesh_ground)
    module.add_simulated_objects(mesh_cloth)

    render_step = 10

    render = Render({
        # 'static_0': mesh.export_for_render(),
        'static_0': mesh_bunny.export_for_render(),
        'static_1': mesh_ground.export_for_render(),
        'simulated_0': mesh_cloth.export_for_render()
        },
        saving=True,
        saving_folder=None,
        subdiv=True,  # todo
        subdiv_time=2, # todo
        ctr_rotate=0.0,  # todo
    )
    sim = Simulation(module, render, max_run_step=5000, render_step=render_step)

    sim.run()