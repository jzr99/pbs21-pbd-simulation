# test
from render.render import Render
import taichi as ti
from lib.mesh import *
from lib.module import *
from lib.simulator import Simulation
from PIL import Image, ImageDraw, ImageFont

def init_sim(render=None, **kwargs):
    mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[1.0, 0.4, 0.2], rescale=0.1, translation=[0, 0.4, 0])
    mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0.35, 0.8, 0.10])
    mesh_cloth.set_gravity_affected(True)
    mesh_cloth.set_wind_affected(True)
    module = Module()
    module.add_static_objects(mesh_sphere)
    module.add_simulated_objects(mesh_cloth)
    if render is None:
        render = Render({'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()})
    sim = Simulation(module, render, **kwargs)
    return sim, {'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()}

def reset_render(render, objs, text, saving_path):

    # render.vis.clear_geometries()
    render.reset_render(objs, text, saving_path)

    # WINDOW_WIDTH = 1920  # change this if needed
    # WINDOW_HEIGHT = 1080  # change this if needed
    # img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color=(255, 255, 255))
    # # fnt = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', 64)
    # d = ImageDraw.Draw(img)
    # font = ImageFont.truetype("./font.ttf", size=30)
    # d.text((50, 50), "Epoch1:", fill=(0, 0, 0), font=font)  # puts text in upper right
    # img.save('./temp.png')
    # im = o3d.io.read_image("./temp.png")
    # render.vis.add_geometry(im)


if __name__ == '__main__':
    ti.init()

    # X, Y, Z 对应关系
    # X，Z：平面坐标，X控制水平方向，从左到右依次递增；Z控制竖直方向，从后到前依次递增。（原点在初始视角平面的左上角）
    # Y：纵向坐标
    # rescale：放大缩小使用；translation：尺度变换后，物体的平移
    # !!!!!!! _large物体由open3d导出，其triangle的索引顺序与小文件的相反，会导致平面法向量计算相反，渲染的颜色面出错
    # mesh_sphere = Mesh(filename='./obj/sphere_large.obj', color=[1.0, 0.4, 0.2], translation=[0, 0.6, 0], reverse_triangle_verts=True)
    # mesh_cloth = Mesh(filename='./obj/cloth_large.obj', color=[0.2, 0.2, 0.2], translation=[0, 1.0, 0], reverse_triangle_verts=True)

    # mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[1.0, 0.4, 0.2], rescale=0.1, translation=[0, 0.4, 0])
    # mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.2, translation=[0.5, 0.8, 0.3])
    # mesh_cloth.set_gravity_affected(True)
    # mesh_cloth.set_wind_affected(True)
    #
    # module = Module()
    # module.add_static_objects(mesh_sphere)
    # module.add_simulated_objects(mesh_cloth)
    # render = Render({'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()})
    # sim = Simulation(module, render)
    sim, objs = init_sim(full_sim=True, total_step=350)
    reset_render(sim.render, objs, " Before Optimization \n Wind Speed {0}".format(str(sim.wind_speed)), 'before')
    sim.run()

    sim, objs = init_sim(render=sim.render)
    reset_render(sim.render, objs, " Epoch {0} \n Wind Speed {1}".format(1, str(sim.wind_speed)), 'optimize')
    iter = 10
    for i in range(iter):
        with ti.Tape(sim.loss):
            sim.run()
            sim.compute_loss()
        grad = sim.wind_speed.grad[None]
        sim.optimize()
        # sim.init_episode()
        old_wind_speed = sim.wind_speed
        # print("wind_speed", sim.wind_speed)
        sim, objs = init_sim(render=sim.render, wind_speed=old_wind_speed)
        reset_render(sim.render, objs, " Epoch {0} \n Wind Speed {1}".format(i+2, str(sim.wind_speed)), 'optimize')
        print("wind_speed", sim.wind_speed)
        # mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[1.0, 0.4, 0.2], rescale=0.1, translation=[0, 0.4, 0])
        # mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.2, translation=[0.5, 0.8, 0.3])
        # mesh_cloth.set_gravity_affected(True)
        # mesh_cloth.set_wind_affected(True)
        # module = Module()
        # module.add_static_objects(mesh_sphere)
        # module.add_simulated_objects(mesh_cloth)
        # sim = Simulation(module, render)
        # sim.wind_speed=old_wind_speed
    sim, objs = init_sim(render=sim.render, wind_speed=old_wind_speed, full_sim=True, total_step=350)
    reset_render(sim.render, objs, " Final Result \n Wind Speed {0}".format(str(sim.wind_speed)), 'after')
    sim.run()