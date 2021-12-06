# import numpy as np
import taichi as ti
# import open3d as o3d
# import utils
from functools import partial


# # TODO replace those things by mesh
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
# # bend_indices = ti.Vector.field(5, float, 3*num_triangles)
#
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

# @ti.func
# def flat2index(flat):
#     j = flat % N
#     i = (flat - j) / N
#     return i, j

# @ti.func
# def calc_angle(p1, p2, p3, p4):
#     x1 = x[flat2index(p1)]
#     x2 = x[flat2index(p2)]
#     x3 = x[flat2index(p3)]
#     x4 = x[flat2index(p4)]
#     # do not consider x1
#     n1 = ti.cross(x2, x3) / ti.cross(x2, x3).norm()
#     n2 = ti.cross(x2, x4) / ti.cross(x2, x4).norm()
#     # TODO clamp the n1*n2 in the range of (-1, 1)
#     return ti.acos(ti.dot(n1, n2))


class BendingConstraints:
    def __init__(self, mesh, bend_factor, solver_iterations):
        # store constraint, each bend_indices has a item of (p1, p2, p3, p4, constrain angle)
        self.bend_indices = ti.Vector.field(5, float, len(mesh._adjacent_triangles))
        self.bend_mask = ti.Vector.field(1, ti.u32, len(mesh._adjacent_triangles))
        self.mesh = mesh
        self.adjacent_triangles = mesh._adjacent_triangles
        # for i in range(3 * num_triangles):
        #     self.bend_indices[i] *= 0
        self.build_constrain()
        self.stiffness = 1.0 - pow(1.0 - bend_factor, 1.0 / solver_iterations)

    # @ti.func
    def calc_angle(self, p1, p2, p3, p4):
        EPSILON = 1e-8
        x1 = self.mesh.vertices[p1]
        x2 = self.mesh.vertices[p2]
        x3 = self.mesh.vertices[p3]
        x4 = self.mesh.vertices[p4]
        # do not consider x1
        n1 = (x2-x1).cross(x3-x1) / (x2-x1).cross(x3-x1).norm()
        n2 = (x2-x1).cross(x4-x1) / (x2-x1).cross(x4-x1).norm()
        # TODO clamp the n1*n2 in the range of (-1, 1)
        d = n1.dot(n2)
        d = max(min(d, 1.0 - EPSILON), -1.0 + EPSILON)
        return ti.acos(d)

    def build_constrain(self):
        for i,(e, adj) in enumerate(self.adjacent_triangles.items()):
            if len(adj) == 2:
                verts_1 = adj[0]
                verts_2 = adj[1]
                p1 = e.v1.p
                p2 = e.v2.p
                # TODO p3 and p4 could be same
                for v in verts_1:
                    if p1!=v.p and p2!=v.p:
                        p3=v.p
                for v in verts_2:
                    if p1!=v.p and p2!=v.p:
                        p4=v.p
                self.bend_indices[i][0] = p1
                self.bend_indices[i][1] = p2
                self.bend_indices[i][2] = p3
                self.bend_indices[i][3] = p4
                self.bend_indices[i][4] = self.calc_angle(p1, p2, p3, p4)
                self.bend_mask[i][0] = 1
            else:
                # print('invalid number of adjacent triangles ',len(adj))
                self.bend_mask[i][0] = 0

                # def build_constrain(self):
    #     for i in range(num_triangles):
    #         if i % 2 == 1:
    #             p1 = indices[i].x
    #             p2 = indices[i].y
    #             p3 = indices[i].z
    #             p4_1 = p2 + 1
    #             p4_2 = p2 - 1
    #             p4_3 = p3 + N
    #             x1, x2 = flat2index(p4_1)
    #             if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
    #                 constrain_id = i * 3
    #                 self.bend_indices[constrain_id][0] = p1
    #                 self.bend_indices[constrain_id][1] = p2
    #                 self.bend_indices[constrain_id][2] = p3
    #                 self.bend_indices[constrain_id][3] = p4_1
    #                 self.bend_indices[constrain_id][4] = calc_angle(p1, p2, p3, p4_1)
    #             x1, x2 = flat2index(p4_3)
    #             if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
    #                 # (p1, p3, p2, p4_3)
    #                 constrain_id = i * 3 + 1
    #                 self.bend_indices[constrain_id][0] = p1
    #                 self.bend_indices[constrain_id][1] = p3
    #                 self.bend_indices[constrain_id][2] = p2
    #                 self.bend_indices[constrain_id][3] = p4_3
    #                 self.bend_indices[constrain_id][4] = calc_angle(p1, p3, p2, p4_3)
    #             x1, x2 = flat2index(p4_2)
    #             if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
    #                 # (p2, p3, p1, p4_2)
    #                 constrain_id = i * 3 + 2
    #                 self.bend_indices[constrain_id][0] = p2
    #                 self.bend_indices[constrain_id][1] = p3
    #                 self.bend_indices[constrain_id][2] = p1
    #                 self.bend_indices[constrain_id][3] = p4_2
    #                 self.bend_indices[constrain_id][4] = calc_angle(p2, p3, p1, p4_2)
    #
    #         if i % 2 == 0:
    #             p1 = indices[i].x
    #             p2 = indices[i].y
    #             p3 = indices[i].z
    #             p4_1 = p2 + 1
    #             p4_2 = p2 - 1
    #             p4_3 = p3 - N
    #             x1, x2 = flat2index(p4_1)
    #             if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
    #                 constrain_id = i * 3
    #                 self.bend_indices[constrain_id][0] = p2
    #                 self.bend_indices[constrain_id][1] = p3
    #                 self.bend_indices[constrain_id][2] = p1
    #                 self.bend_indices[constrain_id][3] = p4_1
    #                 self.bend_indices[constrain_id][4] = calc_angle(p2, p3, p1, p4_1)
    #             x1, x2 = flat2index(p4_2)
    #             if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
    #                 constrain_id = i * 3 + 1
    #                 self.bend_indices[constrain_id][0] = p1
    #                 self.bend_indices[constrain_id][1] = p2
    #                 self.bend_indices[constrain_id][2] = p3
    #                 self.bend_indices[constrain_id][3] = p4_2
    #                 self.bend_indices[constrain_id][4] = calc_angle(p1, p2, p3, p4_2)
    #             x1, x2 = flat2index(p4_3)
    #             if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
    #                 # (p2, p3, p1, p4_2)
    #                 constrain_id = i * 3 + 2
    #                 self.bend_indices[constrain_id][0] = p1
    #                 self.bend_indices[constrain_id][1] = p3
    #                 self.bend_indices[constrain_id][2] = p2
    #                 self.bend_indices[constrain_id][3] = p4_3
    #                 self.bend_indices[constrain_id][4] = calc_angle(p1, p3, p2, p4_3)
    #     # for c in bend_indices:
    #     #     print(bend_indices[c])

    @ti.func
    def project(self):
        EPSILON = 1e-8
        for i in ti.grouped(self.bend_mask):
            # TODO filter zero constrain
            if self.bend_mask[i][0] == 1:
                # print('bend_indices', self.bend_indices[i])
                p1_index, p2_index, p3_index, p4_index, constrain_angle = self.bend_indices[i]
                # p1_index, p2_index, p3_index, p4_index = flat2index(p1_index), flat2index(p2_index), flat2index(p3_index), flat2index(p4_index)
                p1, p2, p3, p4 = self.mesh.estimated_vertices[p1_index], self.mesh.estimated_vertices[p2_index], self.mesh.estimated_vertices[p3_index], self.mesh.estimated_vertices[p4_index]
                # print("p1_old", p1)
                # print("p2_old", p2)
                # print("p3_old", p3)
                # print("p4_old", p4)
                p2_new = p2 - p1
                p3_new = p3 - p1
                p4_new = p4 - p1
                # print("p1", p1)
                # print("p2", p2_new)
                # print("p3", p3_new)
                # print("p4", p4_new)
                p2Xp3 = p2_new.cross(p3_new)
                p2Xp4 = p2_new.cross(p4_new)
                n1 = p2Xp3 / (p2Xp3.norm() + EPSILON)
                n2 = p2Xp4 / (p2Xp4.norm() + EPSILON)
                d = n1.dot(n2)
                # print("d", d)
                # d = max(min(d, ti.Vector([1.0-EPSILON])), ti.Vector([-1.0+EPSILON]))
                d = max(min(d, 1.0 - EPSILON), -1.0 + EPSILON)
                # print("d", d)
                q3 = (p2_new.cross(n2) + d * n1.cross(p2_new)) / (p2Xp3.norm()+EPSILON)
                q4 = (p2_new.cross(n1) + d * n2.cross(p2_new)) / (p2Xp4.norm()+EPSILON)
                q2 = -(p3_new.cross(n2) + d * n1.cross(p3_new)) / (p2Xp3.norm()+EPSILON) - (p4_new.cross(n1) + d * n2.cross(p4_new)) / (
                    p2Xp4.norm()+EPSILON)
                q1 = -q2 - q3 - q4
                # assume all the point has same mass
                # TODO clamp the d in the range of (-1, 1)
                a = ti.sqrt(1.0 - d * d) * (ti.acos(d) - constrain_angle)
                # print('a', a)
                qSum = q1.norm_sqr() + q2.norm_sqr() + q3.norm_sqr() + q4.norm_sqr()
                # qSum = 1000000
                # print('qSum', qSum)
                # print("q1", q1)
                # print("q2", q2)
                # print("q3", q3)
                # print("q4", q4)
                displacements_p1 = (-a * q1) / qSum
                displacements_p2 = (-a * q2) / qSum
                displacements_p3 = (-a * q3) / qSum
                displacements_p4 = (-a * q4) / qSum
                # displacements_p1 = 0
                # displacements_p2 = 0
                # displacements_p3 = 0
                # displacements_p4 = 0
                # update replacement
                # print("bending displacement", displacements_p1, displacements_p2, displacements_p3, displacements_p4)
                # print('stiffness', self.stiffness)
                self.mesh.estimated_vertices[p1_index] += displacements_p1 * self.stiffness
                self.mesh.estimated_vertices[p2_index] += displacements_p2 * self.stiffness
                self.mesh.estimated_vertices[p3_index] += displacements_p3 * self.stiffness
                self.mesh.estimated_vertices[p4_index] += displacements_p4 * self.stiffness