import numpy as np
import taichi as ti
import open3d as o3d
# import utils
from functools import partial


class DistanceConstraintsBuilder:

    def __init__(self, mesh, stiffness_factor, solver_iterations):
        # store constraint, each bend_indices has a item of (p1, p2, p3, p4, constrain angle)
        self.edge_indices = ti.Vector.field(3, ti.float32, len(mesh.edges))
        self.mesh = mesh
        self.edges = mesh.edges
        self.stiffness = 1.0 - pow(1.0 - stiffness_factor, 1.0 / solver_iterations)
        self.build_constrain()

    def build_constrain(self):
        for i, e in enumerate(self.edges):
            self.edge_indices[i][0] = e.v1.p
            self.edge_indices[i][1] = e.v2.p
            self.edge_indices[i][2] = (self.mesh.vertices[e.v1.p] - self.mesh.vertices[e.v2.p]).norm()

    @ti.pyfunc
    def project(self):
        EPSILON = 1e-8
        for i in ti.ndrange(self.edge_indices.shape[0]):
            p1_index, p2_index, constrain_length = self.edge_indices[i]
            # p1_index, p2_index = flat2index(p1_index), flat2index(p2_index)
            p1, p2 = self.mesh.estimated_vertices[int(p1_index)], self.mesh.estimated_vertices[int(p2_index)]
            a = (p1 - p2).norm() - constrain_length
            b = (p1 - p2) / ((p1 - p2).norm() + EPSILON)
            # assume p1 and p2 has same mass
            q1 = (-a * b) * 0.5
            q2 = (a * b) * 0.5
            self.mesh.estimated_vertices[int(p1_index)] += q1 * self.stiffness
            self.mesh.estimated_vertices[int(p2_index)] += q2 * self.stiffness

