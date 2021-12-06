import taichi as ti
from lib.mesh import Vertex
import numpy as np
from lib.module import Module
from lib.distance_constrain import DistanceConstraintsBuilder
from lib.bending_constrain import BendingConstraints
from lib.collision import CollisionConstraints, ray_triangle_intersect


@ti.data_oriented
class Simulation(object):

    def __init__(self, module: Module, render):
        self.module = module
        self.solver_iterations = 4
        # self.iteration_field = ti.Vector.field(1, float, self.solver_iterations)
        self.time_step = 1e-3
        self.gravity = 0.981
        self.wind_speed = 1.0
        self.wind_oscillation = 0
        self.velocity_damping = 0.999
        self.stretch_factor = 0.999
        self.bend_factor = 0.05
        self.collision_threshold = 5e-3
        self.wireframe = False
        self.render = render
        self._mesh_now = None
        self._static_mesh = None
        self._dynamic_mesh = None
        self.collision_constraint = CollisionConstraints(self.module.simulated_objects)
        self.distance_constraint = DistanceConstraintsBuilder(mesh=self.module.simulated_objects[0], stiffness_factor=0.8,
                                                              solver_iterations=self.solver_iterations)
        self.bend_constrain = BendingConstraints(mesh=self.module.simulated_objects[0], bend_factor=self.bend_factor,
                                                 solver_iterations=self.solver_iterations)

    def run(self):
        while True:
            self.wind_oscillation += 0.005
            if not self.render.get_pause():
                self.simulate()
            self.rendering()
            if not self.render.vis.poll_events():
                break
            self.render.vis.update_renderer()

    def simulate(self):
        # velocity and position update under external forces
        for mesh in self.module.simulated_objects:
            self._mesh_now = mesh
            self.simulate_estimate()

        # collision constraints built
        global_offset = 0
        self.collision_constraint.reset()
        for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
            global_offset += dynamic_mesh.num_vertices
            for sta_idx, static_mesh in enumerate(self.module.static_objects):
                self._dynamic_mesh = dynamic_mesh
                self._static_mesh = static_mesh
                self.build_collision_constraints(global_offset)

        # projection
        for _ in range(self.solver_iterations):
            global_offset = 0
            # project by external/collision constraint
            for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
                global_offset += dynamic_mesh.num_vertices
                self._dynamic_mesh = dynamic_mesh
                self.simulate_external_constraint_project(global_offset)
            # project by internal constraint
            self.simulate_internal_constraint_project()

        # calibration (velocity and position update), friction apply
        for mesh in self.module.simulated_objects:
            self._mesh_now = mesh
            self.simulate_calibration_all()

        global_offset = 0
        for dyn_idx, dynamic_mesh in enumerate(self.module.simulated_objects):
            global_offset += dynamic_mesh.num_vertices
            self._dynamic_mesh = dynamic_mesh
            self.simulate_calibration_collision(global_offset)

    @ti.kernel
    def simulate_estimate(self):
        if self._mesh_now.gravity_affected:
            self._mesh_now.apply_impulse(2.0 * self.time_step * ti.Vector([0.0, -self.gravity, 0.0]))
        if self._mesh_now.wind_affected:
            self._mesh_now.apply_impulse(2.0 * self.time_step * ti.Vector([self.wind_speed + np.sin(self.wind_oscillation), 0.0, 0.0]))

        for i in ti.grouped(self._mesh_now.velocities):
            self._mesh_now.estimated_vertices[i] = self._mesh_now.vertices[i] + self.time_step * self._mesh_now.velocities[i]

    @ti.kernel
    def build_collision_constraints(self, global_offset: int):
        """
        Build static collision constraints given dynamic mesh and static mesh.
        :param global_offset: index start of the dynamic mesh vertices in the CollisionConstraintsField.
        """
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
                    self.collision_constraint.add_constraint(global_offset + dyn_ver_idx, surface_norm, entry_point)

    @ti.kernel
    def simulate_internal_constraint_project(self):
        self.distance_constraint.project()
        self.bend_constrain.project()

        # fix the (0, 0) of cloth
        # self._mesh_now.estimated_vertices[0] = self._mesh_now.vertices[0]

    @ti.kernel
    def simulate_external_constraint_project(self, global_offset: int):
        for dyn_ver_idx in ti.grouped(self._dynamic_mesh.estimated_vertices):
            global_index = global_offset + dyn_ver_idx
            est_p = self._dynamic_mesh.estimated_vertices[dyn_ver_idx]
            self.collision_constraint.project(global_index, est_p)

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
