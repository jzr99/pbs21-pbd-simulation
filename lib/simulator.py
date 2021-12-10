import taichi as ti
from lib.mesh import Vertex
import numpy as np
from lib.module import Module
from lib.distance_constrain import DistanceConstraintsBuilder
from lib.bending_constrain import BendingConstraints
from lib.collision import CollisionConstraints, ray_triangle_intersect
from lib.module import *
from render.render import Render
from lib import utils

@ti.data_oriented
class Simulation(object):

    def __init__(self, module: Module, render, **kwargs):

        self.module = module
        self.solver_iterations = 4
        # self.iteration_field = ti.Vector.field(1, float, self.solver_iterations)
        self.time_step = 1e-3
        self.gravity = 0.981
        # self.wind_speed = 1.0
        self.wind_oscillation = 0
        self.velocity_damping = 0.99
        self.stretch_factor = 0.999
        self.bend_factor = 0.001
        self.collision_threshold = 5e-3
        self.self_collision_threshold = 1e-1
        self.cloth_thickness = 1e-2
        self.self_col_factor = 1.0
        self.wireframe = False
        self.render = render
        self._mesh_now = None
        self._static_mesh = None
        self._dynamic_mesh = None
        # self.self_collision = True if len(self.module.simulated_objects) == 1 else False
        self.self_collision = False
        self.collision_constraint = CollisionConstraints(self.module.simulated_objects)
        self.distance_constraint = DistanceConstraintsBuilder(mesh=self.module.simulated_objects[0], stiffness_factor=0.8,
                                                              solver_iterations=self.solver_iterations)
        self.bend_constrain = BendingConstraints(mesh=self.module.simulated_objects[0], bend_factor=self.bend_factor,
                                                 solver_iterations=self.solver_iterations)
        self.min_t = float('inf')
        self.min_idx = 0

        # self._mesh_now.estimated_vertices[0] = self._mesh_now.vertices[0]
        self.init_point = self.module.simulated_objects[0].vertices[0]
        self.loss = ti.field(dtype=ti.f32, needs_grad=True)
        self.wind_speed = ti.Vector.field(n=3, dtype=ti.f32, needs_grad=True)
        self.lr = 10
        ti.root.place(self.loss)
        ti.root.place(self.wind_speed)
        ti.root.lazy_grad()
        self.full_sim = False
        self.__dict__.update(**kwargs)
        if self.self_collision:
            print(f'self collision is supported. ')
        else:
            print(f'self collision is unsupported with {len(self.module.simulated_objects)} dynamic object.')

    def optimize(self):
        self.wind_speed[None][0] -= self.wind_speed.grad[None][0] * self.lr
        self.wind_speed[None][2] -= self.wind_speed.grad[None][2] * self.lr
        print("grad", self.wind_speed.grad[None])
        self.wind_speed.grad[None][0] = 0.0
        self.wind_speed.grad[None][2] = 0.0

    # def init_episode(self):
    #     mesh_sphere = Mesh(filename='./obj/sphere.obj', color=[1.0, 0.4, 0.2], rescale=0.1, translation=[0, 0.4, 0])
    #     mesh_cloth = Mesh(filename='./obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.2, translation=[0.5, 0.8, 0.3])
    #     mesh_cloth.set_gravity_affected(True)
    #     mesh_cloth.set_wind_affected(True)
    #     module = Module()
    #     module.add_static_objects(mesh_sphere)
    #     module.add_simulated_objects(mesh_cloth)
    #     # render = Render({'static_0': mesh_sphere.export_for_render(), 'simulated_0': mesh_cloth.export_for_render()})
    #     self.module = module
    #     # self.render = render
    #     self.solver_iterations = 4
    #     # self.iteration_field = ti.Vector.field(1, float, self.solver_iterations)
    #     self.time_step = 1e-3
    #     self.gravity = 0.981
    #     # self.wind_speed = 1.0
    #     self.wind_oscillation = 0
    #     self.velocity_damping = 0.99
    #     self.stretch_factor = 0.999
    #     self.bend_factor = 0.001
    #     self.collision_threshold = 5e-3
    #     self.self_collision_threshold = 1e-1
    #     self.cloth_thickness = 1e-2
    #     self.self_col_factor = 1.0
    #     self.wireframe = False
    #     # self.render = render
    #     self._mesh_now = None
    #     self._static_mesh = None
    #     self._dynamic_mesh = None
    #     # self.self_collision = True if len(self.module.simulated_objects) == 1 else False
    #     self.self_collision = False
    #     self.collision_constraint = CollisionConstraints(self.module.simulated_objects)
    #     self.distance_constraint = DistanceConstraintsBuilder(mesh=self.module.simulated_objects[0],
    #                                                           stiffness_factor=0.8,
    #                                                           solver_iterations=self.solver_iterations)
    #     self.bend_constrain = BendingConstraints(mesh=self.module.simulated_objects[0], bend_factor=self.bend_factor,
    #                                              solver_iterations=self.solver_iterations)
    #     self.min_t = float('inf')
    #     self.min_idx = 0
    #
    #     # self._mesh_now.estimated_vertices[0] = self._mesh_now.vertices[0]
    #     self.init_point = self.module.simulated_objects[0].vertices[0]
    #     self.loss = ti.field(dtype=ti.f32, needs_grad=True)
    #     self.wind_speed = ti.Vector.field(n=3, dtype=ti.f32, needs_grad=True)
    #     self.lr = 1000
    #     ti.root.place(self.loss)
    #     ti.root.place(self.wind_speed)
    #     ti.root.lazy_grad()

    def run(self):
        # with ti.Tape(self.loss):
        count = 0
        # while count <= 100:
        while True:
            count += 1
            self.wind_oscillation += 0.005
            # if not self.render.get_pause():
            # with ti.Tape(self.loss):
            #     self.simulate()
            #     self.compute_loss()
            if self.full_sim:
                self.full_simulate()
            else:
                self.simulate()
            # print("self.loss", self.loss)
            # self.wind_speed[None][0] -= self.wind_speed.grad[None][0] * self.lr
            # self.wind_speed[None][2] -= self.wind_speed.grad[None][2] * self.lr
            # print("grad", self.wind_speed.grad[None])
            # self.wind_speed.grad[None][0] = 0.0
            # self.wind_speed.grad[None][2] = 0.0
            # print("self.wind_speed", self.wind_speed)
            # print("self.module.simulated_objects[0].estimated_vertices[0]", self.module.simulated_objects[0].estimated_vertices[0])
            # print(count)
            self.rendering()
            if not self.render.vis.poll_events():
                break
            self.render.vis.update_renderer()
            # self.compute_loss()
        # self.wind_speed[None][0] -= self.wind_speed.grad[None][0] * self.lr
        # self.wind_speed[None][2] -= self.wind_speed.grad[None][2] * self.lr
        # print("grad", self.wind_speed.grad[None])
        # self.wind_speed.grad[None][0] = 0.0
        # self.wind_speed.grad[None][2] = 0.0

    @ti.func
    def cal_mean(self):
        mean_point = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.grouped(self.module.simulated_objects[0].estimated_vertices):
            mean_point += (self.module.simulated_objects[0].estimated_vertices[i] / self.module.simulated_objects[0].num_vertices)

        return mean_point

    @ti.kernel
    def compute_loss(self):
        # self.loss[None] = (self.module.simulated_objects[0].estimated_vertices[0] - self.init_point).norm()
        # for i in ti.grouped(self.module.simulated_objects[0].estimated_vertices):
        #     self.module.simulated_objects[0].estimated_vertices[i]
        # self.loss[None] = (self.module.simulated_objects[0].estimated_vertices[0] - self.init_point).norm()
        self.loss[None] = (self.module.simulated_objects[0].estimated_vertices[self.module.simulated_objects[0].num_vertices // 2] - ti.Vector([0, 0.4, 0])).norm()
        # self.loss[None] = self.loss.norm()

    # def new_forward(self):
    #     self.simulate_estimate()
    #     print("center point", self.module.simulated_objects[0].estimated_vertices[self.module.simulated_objects[0].num_vertices // 2])
    #     # self.update_estimate()
    #     # self.forward()

    def full_simulate(self):
        for mesh in self.module.simulated_objects:
            self._mesh_now = mesh
            self.simulate_estimate()
            print("center point", self.module.simulated_objects[0].estimated_vertices[self.module.simulated_objects[0].num_vertices // 2])       # self.new_forward()
                # self.update_estimate()
            # self.compute_loss()
            # self.estimate_all()

        # ----- collision constraints built -----
        global_offset = 0
        self.collision_constraint.reset()
        # static collision
        # with ti.Tape(self.loss):
        # static collision
        for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
            global_offset += dynamic_mesh.num_vertices
            for sta_idx, static_mesh in enumerate(self.module.static_objects):
                self._dynamic_mesh = dynamic_mesh
                self._static_mesh = static_mesh
                # update the bounding box of static object for broad collision detection
                self._static_mesh.bounding_box.update_bounding_box(self._static_mesh.vertices)
                self.build_collision_constraints(global_offset)

            # dynamic self collision
        if self.self_collision:
            self._dynamic_mesh = self.module.simulated_objects[0]
            self.build_self_collision_constraint()

        # ----------------------------------------

        # ------------- projection ---------------
        # with ti.Tape(self.loss):
        for _ in ti.static(range(self.solver_iterations)):
            global_offset = 0
            # project by external/collision constraint
            for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
                global_offset += dynamic_mesh.num_vertices
                self._dynamic_mesh = dynamic_mesh
                self.simulate_external_constraint_project(global_offset)

            # project by internal self collision constraint
            if self.self_collision:
                self._dynamic_mesh = self.module.simulated_objects[0]
                self.simulate_self_constraint_project()
            # project by internal constraint
            self.simulate_internal_constraint_project()
        #
        #     # with ti.Tape(self.loss):
        #     #     self.simulate_internal_constraint_project()
        #     #     self.compute_loss()
        # ----------------------------------------

        # ----- calibration (velocity and position update), friction apply -----
        for mesh in self.module.simulated_objects:
            self._mesh_now = mesh
            self.simulate_calibration_all()


        global_offset = 0
        for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
            global_offset += dynamic_mesh.num_vertices
            self._dynamic_mesh = dynamic_mesh
            self.simulate_calibration_collision(global_offset)
        # -----------------------------------------------------------------------

    def simulate(self):
        # velocity and position update under external forces

        for mesh in self.module.simulated_objects:
            self._mesh_now = mesh
            self.simulate_estimate()
            print("center point", self.module.simulated_objects[0].estimated_vertices[self.module.simulated_objects[0].num_vertices // 2])       # self.new_forward()
                # self.update_estimate()
            # self.compute_loss()
            # self.estimate_all()

        # ----- collision constraints built -----
        # global_offset = 0
        # self.collision_constraint.reset()
        # # static collision
        # # with ti.Tape(self.loss):
        # # static collision
        # for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
        #     global_offset += dynamic_mesh.num_vertices
        #     for sta_idx, static_mesh in enumerate(self.module.static_objects):
        #         self._dynamic_mesh = dynamic_mesh
        #         self._static_mesh = static_mesh
        #         # update the bounding box of static object for broad collision detection
        #         self._static_mesh.bounding_box.update_bounding_box(self._static_mesh.vertices)
        #         self.build_collision_constraints(global_offset)

            # dynamic self collision
        # if self.self_collision:
        #     self._dynamic_mesh = self.module.simulated_objects[0]
        #     self.build_self_collision_constraint()

        # ----------------------------------------

        # ------------- projection ---------------
        # with ti.Tape(self.loss):
        for _ in ti.static(range(self.solver_iterations)):
            # global_offset = 0
            # # project by external/collision constraint
            # for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
            #     global_offset += dynamic_mesh.num_vertices
            #     self._dynamic_mesh = dynamic_mesh
            #     self.simulate_external_constraint_project(global_offset)

            # project by internal self collision constraint
            # if self.self_collision:
            #     self._dynamic_mesh = self.module.simulated_objects[0]
            #     self.simulate_self_constraint_project()
            # project by internal constraint
            self.simulate_internal_constraint_project()
        #
        #     # with ti.Tape(self.loss):
        #     #     self.simulate_internal_constraint_project()
        #     #     self.compute_loss()
        # ----------------------------------------

        # ----- calibration (velocity and position update), friction apply -----
        for mesh in self.module.simulated_objects:
            self._mesh_now = mesh
            self.simulate_calibration_all()


        # global_offset = 0
        # for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
        #     global_offset += dynamic_mesh.num_vertices
        #     self._dynamic_mesh = dynamic_mesh
        #     self.simulate_calibration_collision(global_offset)
        # -----------------------------------------------------------------------

    # @ti.kernel
    # def estimate_all(self):
    #     self.simulate_estimate()
    #     self.update_estimate()



    @ti.kernel
    def simulate_estimate(self):
        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.velocities[i] = 2.0 * self.time_step * ti.Vector([0.0, -self.gravity, 0.0]) + self._mesh_now.velocities[i]
            self._mesh_now.velocities[i] = (2.0 * self.time_step * ti.Vector([self.wind_speed[None][0] + np.sin(self.wind_oscillation), 0.0, self.wind_speed[None][2] + np.sin(self.wind_oscillation)])) + self._mesh_now.velocities[i]
            self._mesh_now.estimated_vertices[i] = self._mesh_now.vertices[i] + self.time_step * self._mesh_now.velocities[i]
        # for i in ti.grouped(self._mesh_now.velocities):
        #     self._mesh_now.estimated_vertices[i] = self._mesh_now.vertices[i] + self.time_step * self._mesh_now.velocities[i]

        # for i in ti.ndrange(*self._mesh_now.velocities.shape):
        #     # if self._mesh_now.gravity_affected:
        #     #     self._mesh_now.apply_impulse(2.0 * self.time_step * ti.Vector([0.0, -self.gravity, 0.0]))
        #     if self._mesh_now.wind_affected:
        #         self._mesh_now.apply_impulse(2.0 * self.time_step * ti.Vector([self.wind_speed[None] + np.sin(self.wind_oscillation), 0.0, 0.0]))
                # self._mesh_now.velocities[i] += (2.0 * self.time_step * ti.Vector([self.wind_speed[None] + np.sin(self.wind_oscillation), 0.0, 0.0]))

    # @ti.func
    # def apply_impose(self):


    @ti.kernel
    def update_estimate(self):
        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.estimated_vertices[i] = self._mesh_now.vertices[i] + self.time_step * self._mesh_now.velocities[i]



    @ti.pyfunc
    def fucking_constraint(self, ray: ti.template(), ray_origin: ti.template(), ray_direction: ti.template()):
        min_t = float('inf')
        min_idx = 0
        for static_triangle_idx in range(self._static_mesh.triangle.shape[0]):
            v0_idx, v1_idx, v2_idx = self._static_mesh.triangle[static_triangle_idx]
            v0 = self._static_mesh.vertices[v0_idx]
            v1 = self._static_mesh.vertices[v1_idx]
            v2 = self._static_mesh.vertices[v2_idx]
            t = ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2)
            # collision detected
            if t[0] > 0 and t[0] * 0.5 <= ray.norm() + self.collision_threshold:
                # compute surface norm: make sure it is in the reverse direction of ray
                surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                if surface_norm.dot(ray_direction) > 0:
                    surface_norm = - surface_norm
                entry_point = ray_origin + (t[0] - self.collision_threshold) * ray_direction
                # only build collision constraint with the closest triangle
                if t[0] < min_t:
                    min_t = t[0]
                    min_idx = static_triangle_idx
        return min_t, min_idx

    @ti.kernel
    def build_collision_constraints(self, global_offset: int):
        """
        Build static collision constraints given dynamic mesh and static mesh.
        :param global_offset: index start of the dynamic mesh vertices in the CollisionConstraintsField.
        """
        # for dyn_ver_idx in ti.ndrange(*self._dynamic_mesh.estimated_vertices.shape):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            ray = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - self._dynamic_mesh.vertices[dyn_ver_idx])  # todo
            ray_origin = self._dynamic_mesh.vertices[dyn_ver_idx]
            ray_direction = ray.normalized()

            # broad collision detection
            if self._static_mesh.bounding_box.intersect(ray_origin, ray_direction):
                # narrow collision detection
                min_t = float('inf')
                min_idx = 0
                min_t, min_idx = self.fucking_constraint(ray, ray_origin, ray_direction)

                if min_t < float('inf'):
                    v0_idx, v1_idx, v2_idx = self._static_mesh.triangle[min_idx]
                    v0 = self._static_mesh.vertices[v0_idx]
                    v1 = self._static_mesh.vertices[v1_idx]
                    v2 = self._static_mesh.vertices[v2_idx]
                    surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                    if surface_norm.dot(ray_direction) > 0:
                        surface_norm = - surface_norm
                    entry_point = ray_origin + (min_t - self.collision_threshold) * ray_direction
                    self.collision_constraint.add_constraint(global_offset + dyn_ver_idx, surface_norm, entry_point)

    @ti.kernel
    def build_self_collision_constraint(self):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            ray = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - self._dynamic_mesh.vertices[dyn_ver_idx])  # todo
            ray_origin = self._dynamic_mesh.vertices[dyn_ver_idx]
            ray_end = self._dynamic_mesh.estimated_vertices[dyn_ver_idx]
            ray_direction = ray.normalized()

            min_t = float('inf')
            min_dyn_triangle_idx = 0
            min_v0_idx, min_v1_idx, min_v2_idx = 0, 0, 0
            for dyn_triangle_idx in range(self._dynamic_mesh.triangle.shape[0]):
                v0_idx, v1_idx, v2_idx = self._dynamic_mesh.triangle[dyn_triangle_idx]
                # if the vertice on the triangle mesh, skip
                if not(v0_idx == dyn_ver_idx[0] or v1_idx == dyn_ver_idx[0] or v2_idx == dyn_ver_idx[0]):
                    v0 = self._dynamic_mesh.estimated_vertices[v0_idx]
                    v1 = self._dynamic_mesh.estimated_vertices[v1_idx]
                    v2 = self._dynamic_mesh.estimated_vertices[v2_idx]
                    # if triangle and target vertice too far, skip
                    if not ((ray_origin - 1 / 3 * (v0 + v1 + v2)).norm() > self.self_collision_threshold):
                        # narrow detection
                        t = ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2)
                        if t[0] > 0:
                            # compute surface norm: make sure it is in the reverse direction of ray
                            surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                            if surface_norm.dot(ray_direction) > 0:
                                surface_norm = - surface_norm
                                tmp = v2_idx
                                v2_idx = v1_idx
                                v1_idx = tmp
                            # self collision constrain satisfied
                            if (ray_end - v0).dot(surface_norm) < self.cloth_thickness and t[0] < min_t:
                                min_t = t[0]
                                min_dyn_triangle_idx = dyn_triangle_idx
                                min_v0_idx = v0_idx
                                min_v1_idx = v1_idx
                                min_v2_idx = v2_idx

            # add self constraint if has
            if min_t < float('inf'):
                v0 = self._dynamic_mesh.estimated_vertices[min_v0_idx]
                v1 = self._dynamic_mesh.estimated_vertices[min_v1_idx]
                v2 = self._dynamic_mesh.estimated_vertices[min_v2_idx]
                surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                entry_point = ray_origin + min_t * ray_direction
                # print(dyn_ver_idx)
                # print('ray origin', self._dynamic_mesh.vertices[dyn_ver_idx])
                # print('est position', self._dynamic_mesh.estimated_vertices[dyn_ver_idx])
                # print('velocity', self._dynamic_mesh.velocities[dyn_ver_idx])
                # print('p1', self._dynamic_mesh.estimated_vertices[min_v0_idx])
                # print('p2', self._dynamic_mesh.estimated_vertices[min_v1_idx])
                # print('p3', self._dynamic_mesh.estimated_vertices[min_v2_idx])
                self.collision_constraint.add_self_constraint(dyn_ver_idx, min_v0_idx, min_v1_idx, min_v2_idx, surface_norm, entry_point)

    @ti.kernel
    def simulate_internal_constraint_project(self):
        self.distance_constraint.project()
        # self.bend_constrain.project()

        # fix the (0, 0) of cloth
        # self._mesh_now.estimated_vertices[0] = self._mesh_now.vertices[0]

    @ti.kernel
    def simulate_external_constraint_project(self, global_offset: int):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            global_index = global_offset + dyn_ver_idx
            self.collision_constraint.project(global_index, self._dynamic_mesh.estimated_vertices[dyn_ver_idx])

    def simulate_self_constraint_project(self):
        # with ti.Tape(self.collision_constraint.self_collision_sum):
        #     self.compute_valid_self_collision_sum()

        self.project_self_collision()

    @ti.kernel
    def compute_valid_self_collision_sum(self):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            # no self collision in the collision detection phase
            if self.collision_constraint.has_self_constraint[dyn_ver_idx] == 1:
                v0_idx, v1_idx, v2_idx = self.collision_constraint.self_other_vertices_idx[dyn_ver_idx]
                v0 = self._dynamic_mesh.estimated_vertices[v0_idx]
                v1 = self._dynamic_mesh.estimated_vertices[v1_idx]
                v2 = self._dynamic_mesh.estimated_vertices[v2_idx]
                surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                constraint = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - v0).dot(surface_norm) - self.cloth_thickness

                # self collision already be solved
                if constraint < 0:
                    self.collision_constraint.self_collision_sum[None] += constraint

    @ti.kernel
    def project_self_collision(self):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            # no self collision in the collision detection phase
            if not self.collision_constraint.has_self_constraint[dyn_ver_idx] == 0:
                v0_idx, v1_idx, v2_idx = self.collision_constraint.self_other_vertices_idx[dyn_ver_idx]
                v0 = self._dynamic_mesh.estimated_vertices[v0_idx]
                v1 = self._dynamic_mesh.estimated_vertices[v1_idx]
                v2 = self._dynamic_mesh.estimated_vertices[v2_idx]
                surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                constraint = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - v0).dot(surface_norm) - self.cloth_thickness

                # self collision already be solved
                if constraint < 0:
                    # project by gradient - failed
                    # q_grad = self._dynamic_mesh.estimated_vertices.grad[dyn_ver_idx]
                    # v0_grad = self._dynamic_mesh.estimated_vertices.grad[v0_idx]
                    # v1_grad = self._dynamic_mesh.estimated_vertices.grad[v1_idx]
                    # v2_grad = self._dynamic_mesh.estimated_vertices.grad[v2_idx]
                    # s = constraint / (q_grad.norm() ** 2 + v0_grad.norm() ** 2 + v1_grad.norm() ** 2 + v2_grad.norm() ** 2)
                    # s = s * (1 - (1 - self.self_col_factor) ** (1 / self.solver_iterations))
                    # self._dynamic_mesh.estimated_vertices[dyn_ver_idx] += - s * q_grad
                    # self._dynamic_mesh.estimated_vertices[v0_idx] += -s * v0_grad
                    # self._dynamic_mesh.estimated_vertices[v1_idx] += -s * v1_grad
                    # self._dynamic_mesh.estimated_vertices[v2_idx] += -s * v2_grad

                    # project by moving - failed
                    # entry_to_p = self._dynamic_mesh.estimated_vertices[dyn_ver_idx] \
                    #              - self.collision_constraint.self_collision_entry_point[dyn_ver_idx]
                    # self._dynamic_mesh.estimated_vertices[dyn_ver_idx] += entry_to_p.dot(surface_norm) * entry_to_p.normalized()

                    # standard projection solver
                    n = surface_norm
                    p1 = v1 - v0
                    p2 = v2 - v0
                    q = self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - v0
                    # derivative of p1
                    tmp_p1 = n @ (n.cross(p2).transpose()) + [[0, p2.z, -p2.y], [-p2.z, 0, p2.x], [p2.y, -p2.x, 0]]  # 3 x 3
                    dc_dp1 = (q.transpose() @ tmp_p1).transpose() / p1.cross(p2).norm()  # 3 x 1
                    # derivative of p2
                    tmp_p2 = n @ (n.cross(p1).transpose()) + [[0, p1.z, -p1.y], [-p1.z, 0, p1.x], [p1.y, -p1.x, 0]]  # 3 x 3
                    dc_dp2 = - (q.transpose() @ tmp_p2).transpose() / p1.cross(p2).norm()  # 3 x 1
                    # derivative of q
                    dc_dq = p1.cross(p2).normalized()
                    # derivative of p0
                    dc_dp0 = - dc_dp1 - dc_dp2 - dc_dq

                    # projection
                    s = constraint / (dc_dq.norm() ** 2 + dc_dp0.norm() ** 2 + dc_dp1.norm() ** 2 + dc_dp2.norm() ** 2)
                    s = s * (1 - (1 - self.self_col_factor) ** (1 / self.solver_iterations))
                    self._dynamic_mesh.estimated_vertices[v0_idx] += -s * dc_dp0
                    self._dynamic_mesh.estimated_vertices[v1_idx] += -s * dc_dp1
                    self._dynamic_mesh.estimated_vertices[v2_idx] += -s * dc_dp2
                    self._dynamic_mesh.estimated_vertices[dyn_ver_idx] += -s * dc_dq

    @ti.kernel
    def simulate_calibration_all(self):
        # update velocities and positions
        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.velocities[i] = (self._mesh_now.estimated_vertices[i] - self._mesh_now.vertices[i]) / self.time_step
            self._mesh_now.velocities[i] = self._mesh_now.velocities[i] * self.velocity_damping
            self._mesh_now.vertices[i] = self._mesh_now.estimated_vertices[i]

    @ti.kernel
    def simulate_calibration_collision(self, global_offset: int):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            global_index = global_offset + dyn_ver_idx
            v = self._dynamic_mesh.velocities[dyn_ver_idx]
            self.collision_constraint.calibrate_colliding_vertices(global_index, v)

    def rendering(self):
        update_dict = dict()
        for i, mesh in enumerate(self.module.simulated_objects):
            update_dict['simulated_{}'.format(i)] = mesh.export_for_render()
        for i, mesh in enumerate(self.module.static_objects):
            update_dict['static_{}'.format(i)] = mesh.export_for_render()
        self.render.update(update_dict)
