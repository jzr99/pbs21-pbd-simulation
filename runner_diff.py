# test
from render.render import Render
import taichi as ti
from lib.mesh import *
from lib.module import *
from lib.simulator import Simulation
from PIL import Image, ImageDraw, ImageFont

def init_sim(render=None, **kwargs):
    mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[1.0, 0.4, 0.2], rescale=0.07, translation=[0, 0.4, 0])
    mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.06, translation=[0.30, 0.8, 0.02])
    mesh_cloth.set_gravity_affected(True)
    mesh_cloth.set_wind_affected(True)
    module = Module()
    module.add_static_objects(mesh_sphere)
    module.add_simulated_objects(mesh_cloth)
    if render is None:
        render = Render({'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()})
    sim = Simulation(module, render, **kwargs)
    return sim, {'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()}


if __name__ == '__main__':
    ti.init()

    sim, objs = init_sim(full_sim=True, total_step=800)
    sim.render.reset_render(objs, " Before Optimization \n Wind Speed {0}".format(str(sim.wind_speed)), 'before', update_times=0)
    sim.run()
    sim, objs = init_sim(render=sim.render)
    sim.render.reset_render(objs, " Epoch {0} \n Wind Speed {1}".format(1, str(sim.wind_speed)), 'optimize', update_times=0)
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
        sim.render.reset_render(objs, " Epoch {0} \n Wind Speed {1}".format(i+2, str(sim.wind_speed)), 'optimize')
        print("wind_speed", sim.wind_speed)
    sim, objs = init_sim(sim.render, wind_speed=old_wind_speed, full_sim=True, total_step=800)
    sim.render.reset_render(objs, " Final Result \n Wind Speed {0}".format(str(sim.wind_speed)), 'after', update_times=0)
    sim.run()