import numpy as np
import taichi as ti
import open3d as o3d
# import utils
from functools import partial


# TODO replace those things by mesh
# N = 128
# W = 2
# L = W / N
# x = ti.Vector.field(3, float, (N, N))
# v = ti.Vector.field(3, float, (N, N))
# fint = ti.Vector.field(3, float, (N, N))
# fext = ti.Vector.field(3, float, (N, N))
# fdamp = ti.Vector.field(3, float, (N, N))
#
# num_triangles = (N - 1) * (N - 1) * 2
# indices = ti.Vector.field(3, int, num_triangles)
#
# links = ti.Vector.field(2, int, 8)
# links.from_numpy(np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]]))
# links_end = 8 # take the first 8 elements of links


# for i, j in ti.ndrange(N, N):
#     x[i, j] *= 0
#     v[i, j] *= 0
#     fint[i, j] *= 0
#     fext[i, j] *= 0
#     fdamp[i, j] *= 0
#
# for i, j in ti.ndrange(N, N):
#     x[i, j] = ti.Vector(
#         [(i + 0.5) * L - 0.5 * W, (j + 0.5) * L / ti.sqrt(2) + 1.0, (N - j) * L / ti.sqrt(2) - 0.4 * W]
#     )
#
#     if i < N - 1 and j < N - 1:
#         tri_id = ((i * (N - 1)) + j) * 2
#         indices[tri_id].x = i * N + j
#         indices[tri_id].y = (i + 1) * N + j
#         indices[tri_id].z = i * N + (j + 1)
#
#         tri_id += 1
#         indices[tri_id].x = (i + 1) * N + j + 1
#         indices[tri_id].y = i * N + (j + 1)
#         indices[tri_id].z = (i + 1) * N + j
#
#
# @ti.func
# def flat2index(flat):
#     j = flat % N
#     i = (flat - j) / N
#     return i, j
#
# @ti.func
# def index2flat(i, j):
#     return i * N + j

class DistanceConstraintsBuilder:

    def __init__(self, mesh, stiffness_factor, solver_iterations):
        # store constraint, each bend_indices has a item of (p1, p2, p3, p4, constrain angle)
        self.edge_indices = ti.Vector.field(3, float, len(mesh.edges))
        # for i in range(8 * N * N):
        #     self.edge_indices[i] *= 0
        self.mesh = mesh
        self.edges = mesh.edges
        self.stiffness = 1.0 - pow(1.0 - stiffness_factor, 1.0 / solver_iterations)
        self.build_constrain()


    def build_constrain(self):
        for i, e in enumerate(self.edges):
            self.edge_indices[i][0] = e.v1.p
            self.edge_indices[i][1] = e.v2.p
            self.edge_indices[i][2] = (self.mesh.vertices[e.v1.p] - self.mesh.vertices[e.v2.p]).norm()


    # def build_constrain(self):
    #     for i in ti.grouped(x):
    #         for k in range(links_end):
    #             d = links[k]
    #             index = min(max(i + d, 0), ti.Vector([N - 1, N - 1]))
    #             flat_index = index2flat(index[0], index[1])
    #             self.edge_indices[flat_index + k][0] = index2flat(i[0], i[1])
    #             self.edge_indices[flat_index + k][1] = flat_index
    #             # without consider length
    #             self.edge_indices[flat_index + k][2] = L * float(d).norm()
    @ti.func
    def project(self):
        EPSILON = 1e-8
        for i in ti.grouped(self.edge_indices):
            p1_index, p2_index, constrain_length = self.edge_indices[i]
            # p1_index, p2_index = flat2index(p1_index), flat2index(p2_index)
            p1, p2 = self.mesh.estimated_vertices[p1_index], self.mesh.estimated_vertices[p2_index]
            a = (p1 - p2).norm() - constrain_length
            b = (p1 - p2) / ((p1 - p2).norm() + EPSILON)
            # assume p1 and p2 has same mass
            q1 = (-a * b) * 0.5
            q2 = (a * b) * 0.5
            self.mesh.estimated_vertices[p1_index] += q1 * self.stiffness
            self.mesh.estimated_vertices[p2_index] += q2 * self.stiffness

# class DistanceConstraints:
#     def __init__(self, p1, p2, distance):
#         self.p1 = p1
#         self.p2 = p2
#         self.distance = distance
#
#     def project(self):

