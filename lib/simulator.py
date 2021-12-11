import taichi as ti
from lib.mesh import Vertex
import numpy as np
from lib.module import Module
from lib.distance_constrain import DistanceConstraintsBuilder
from lib.bending_constrain import BendingConstraints
from lib.collision import CollisionConstraints, ray_triangle_intersect


@ti.data_oriented
class Simulation(object):

    def __init__(self, module: Module, render, max_run_step=10000):
        self.module = module
        self.max_run_step = max_run_step
        self.solver_iterations = 4
        # self.iteration_field = ti.Vector.field(1, float, self.solver_iterations)
        self.time_step = 1e-3
        self.gravity = 0.981
        self.wind_speed = 0.3
        self.wind_oscillation = 0
        self.velocity_damping = 0.999
        self.stretch_factor = 0.999
        self.bend_factor = 0.001
        self.collision_threshold = 1e-2
        self.self_collision_threshold = 1e-1
        self.cloth_thickness = 2e-2
        self.self_col_factor = 1.0
        self.restitution = 0.2
        self.friction = 0.1
        self.wireframe = False
        self.render = render
        self._mesh_now = None
        self._static_mesh = None
        self._dynamic_mesh = None
        self._ground_mesh = None
        # self.self_collision = True if len(self.module.simulated_objects) == 1 else False
        self.self_collision = False
        self.collision_constraint = CollisionConstraints(self.module.simulated_objects, self.friction, self.restitution)
        self.distance_constraint = DistanceConstraintsBuilder(mesh=self.module.simulated_objects[0], stiffness_factor=0.8,
                                                              solver_iterations=self.solver_iterations)
        self.bend_constrain = BendingConstraints(mesh=self.module.simulated_objects[0], bend_factor=self.bend_factor,
                                                 solver_iterations=self.solver_iterations)
        if self.self_collision:
            print(f'self collision is supported. ')
        else:
            print(f'self collision is unsupported with {len(self.module.simulated_objects)} dynamic object.')

    def run(self):
        count = 0
        while True:
            count += 1
            self.wind_oscillation += 0.01
            if not self.render.get_pause() and count < self.max_run_step:
                self.simulate()
            self.rendering()
            if count == 10000:
                # file format p1_index p2_index p3_index p4_index angle
                self.bend_constrain.write_angle_csv('./bending_angle.csv')
            if not self.render.vis.poll_events():
                break
            self.render.vis.update_renderer()

    def simulate(self):
        self._mesh_now = self.module.simulated_objects[0]
        self._dynamic_mesh = self.module.simulated_objects[0]
        self._static_mesh = self.module.static_objects[0]
        self._ground_mesh = self.module.static_objects[1]
        # velocity and position update under external forces
        self.simulate_estimate()

        # ----- collision constraints built -----
        self.collision_constraint.reset()
        # static collision
        self.build_collision_constraints()
        # dynamic self collision
        if self.self_collision:
            # self._dynamic_mesh = self.module.simulated_objects[0]
            self.build_self_collision_constraint()
        # ----------------------------------------

        # ------------- projection ---------------
        for _ in range(self.solver_iterations):
            # project by external/collision constraint
            self.simulate_external_constraint_project()
            # project by internal self collision constraint
            if self.self_collision:
                self.simulate_self_constraint_project()
            # project by internal constraint
            self.simulate_internal_constraint_project()
        # ----------------------------------------

        # ----- calibration (velocity and position update), friction apply -----
        self.simulate_calibration_all()
        self.simulate_calibration_collision()
        self.simulate_velocity_damping()
    # -----------------------------------------------------------------------

    @ti.kernel
    def simulate_estimate(self):
        if self._mesh_now.gravity_affected:
            self._mesh_now.apply_impulse(2.0 * self.time_step * ti.Vector([0.0, -self.gravity, 0.0]))
        if self._mesh_now.wind_affected:
            self._mesh_now.apply_impulse(2.0 * self.time_step * ti.Vector([0.0, 0.0, self.wind_speed + self.wind_speed * np.sin(self.wind_oscillation)]))

        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.estimated_vertices[i] = self._mesh_now.vertices[i] + self.time_step * self._mesh_now.velocities[i]

    @ti.kernel
    def build_collision_constraints(self):
        """
        Build static collision constraints given dynamic mesh and static mesh.
        :param global_offset: index start of the dynamic mesh vertices in the CollisionConstraintsField.
        """
        # for static object
        # update the bounding box of static object for broad collision detection
        self._static_mesh.bounding_box.update_bounding_box(self._static_mesh.vertices)

        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            ray = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - self._dynamic_mesh.vertices[dyn_ver_idx])  # todo
            ray_origin = self._dynamic_mesh.vertices[dyn_ver_idx]
            ray_direction = ray.normalized()

            # broad collision detection
            if not self._static_mesh.bounding_box.intersect(ray_origin, ray_direction):
                continue

            # narrow collision detection
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

            if min_t < float('inf'):
                v0_idx, v1_idx, v2_idx = self._static_mesh.triangle[min_idx]
                v0 = self._static_mesh.vertices[v0_idx]
                v1 = self._static_mesh.vertices[v1_idx]
                v2 = self._static_mesh.vertices[v2_idx]
                surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                if surface_norm.dot(ray_direction) > 0:
                    surface_norm = - surface_norm
                entry_point = ray_origin + (min_t - self.collision_threshold) * ray_direction
                self.collision_constraint.add_constraint(dyn_ver_idx, surface_norm, entry_point)

        # for ground plane
        # update the bounding box of static object for broad collision detection
        self._ground_mesh.bounding_box.update_bounding_box(self._ground_mesh.vertices)

        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            ray = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - self._dynamic_mesh.vertices[dyn_ver_idx])  # todo
            ray_origin = self._dynamic_mesh.vertices[dyn_ver_idx]
            ray_direction = ray.normalized()

            # broad collision detection
            if not self._ground_mesh.bounding_box.intersect(ray_origin, ray_direction):
                continue

            # narrow collision detection
            min_t = float('inf')
            min_idx = 0
            for static_triangle_idx in range(self._ground_mesh.triangle.shape[0]):
                v0_idx, v1_idx, v2_idx = self._ground_mesh.triangle[static_triangle_idx]
                v0 = self._ground_mesh.vertices[v0_idx]
                v1 = self._ground_mesh.vertices[v1_idx]
                v2 = self._ground_mesh.vertices[v2_idx]
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

            if min_t < float('inf'):
                v0_idx, v1_idx, v2_idx = self._ground_mesh.triangle[min_idx]
                v0 = self._ground_mesh.vertices[v0_idx]
                v1 = self._ground_mesh.vertices[v1_idx]
                v2 = self._ground_mesh.vertices[v2_idx]
                surface_norm = (v1 - v0).cross(v2 - v0).normalized()
                if surface_norm.dot(ray_direction) > 0:
                    surface_norm = - surface_norm
                entry_point = ray_origin + (min_t - self.collision_threshold) * ray_direction
                self.collision_constraint.add_constraint(dyn_ver_idx, surface_norm, entry_point)

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
                if v0_idx == dyn_ver_idx[0] or v1_idx == dyn_ver_idx[0] or v2_idx == dyn_ver_idx[0]:
                    continue
                v0 = self._dynamic_mesh.estimated_vertices[v0_idx]
                v1 = self._dynamic_mesh.estimated_vertices[v1_idx]
                v2 = self._dynamic_mesh.estimated_vertices[v2_idx]
                # if triangle and target vertice too far, skip
                if (ray_origin - 1 / 3 * (v0 + v1 + v2)).norm() > self.self_collision_threshold:
                    continue
                # if vertices and triangle are too close based on initial location, skip
                v0_init, v1_init, v2_init = self._dynamic_mesh.initial_vertices[v0_idx], \
                                            self._dynamic_mesh.initial_vertices[v1_idx], \
                                            self._dynamic_mesh.initial_vertices[v2_idx]
                if (self._dynamic_mesh.initial_vertices[dyn_ver_idx] - 1 / 3 * (v0_init + v1_init + v2_init)).norm() < self.self_collision_threshold:
                    continue

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
        self.bend_constrain.project()

        # fix the (0, 0) of cloth
        # self._mesh_now.estimated_vertices[406] = self._mesh_now.vertices[406]
        # self._mesh_now.estimated_vertices[431] = self._mesh_now.vertices[431]
        # self._mesh_now.estimated_vertices[499] = self._mesh_now.vertices[499]
        # self._mesh_now.estimated_vertices[524] = self._mesh_now.vertices[524]

    @ti.kernel
    def simulate_external_constraint_project(self):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            global_index = dyn_ver_idx
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
            if self.collision_constraint.has_self_constraint[dyn_ver_idx] == 0:
                continue

            v0_idx, v1_idx, v2_idx = self.collision_constraint.self_other_vertices_idx[dyn_ver_idx]
            v0 = self._dynamic_mesh.estimated_vertices[v0_idx]
            v1 = self._dynamic_mesh.estimated_vertices[v1_idx]
            v2 = self._dynamic_mesh.estimated_vertices[v2_idx]
            surface_norm = (v1 - v0).cross(v2 - v0).normalized()
            constraint = (self._dynamic_mesh.estimated_vertices[dyn_ver_idx] - v0).dot(surface_norm) - self.cloth_thickness

            # self collision already be solved
            if constraint >= 0:
                continue

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
            s *= self.self_col_factor
            self._dynamic_mesh.estimated_vertices[v0_idx] += -s * dc_dp0
            self._dynamic_mesh.estimated_vertices[v1_idx] += -s * dc_dp1
            self._dynamic_mesh.estimated_vertices[v2_idx] += -s * dc_dp2
            self._dynamic_mesh.estimated_vertices[dyn_ver_idx] += -s * dc_dq

    @ti.kernel
    def simulate_calibration_all(self):
        # update velocities and positions
        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.velocities[i] = (self._mesh_now.estimated_vertices[i] - self._mesh_now.vertices[i]) / self.time_step
            # self._mesh_now.velocities[i] = self._mesh_now.velocities[i] * self.velocity_damping
            self._mesh_now.vertices[i] = self._mesh_now.estimated_vertices[i]

    @ti.kernel
    def simulate_calibration_collision(self):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            global_index = dyn_ver_idx
            v = self._dynamic_mesh.velocities[dyn_ver_idx]
            self.collision_constraint.calibrate_colliding_vertices(global_index, v)

    @ti.kernel
    def simulate_velocity_damping(self):
        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.velocities[i] = self._mesh_now.velocities[i] * self.velocity_damping

    def rendering(self):
        update_dict = dict()
        for i, mesh in enumerate(self.module.simulated_objects):
            update_dict['simulated_{}'.format(i)] = mesh.export_for_render()
        for i, mesh in enumerate(self.module.static_objects):
            update_dict['static_{}'.format(i)] = mesh.export_for_render()
        self.render.update(update_dict)
