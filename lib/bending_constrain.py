import numpy as np
import taichi as ti
import open3d as o3d
import utils
from functools import partial


# TODO replace those things by mesh
N = 128
W = 2
L = W / N
x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))
fint = ti.Vector.field(3, float, (N, N))
fext = ti.Vector.field(3, float, (N, N))
fdamp = ti.Vector.field(3, float, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.Vector.field(3, int, num_triangles)
bend_indices = ti.Vector.field(5, float, 3*num_triangles)

for i, j in ti.ndrange(N, N):
    x[i, j] *= 0
    v[i, j] *= 0
    fint[i, j] *= 0
    fext[i, j] *= 0
    fdamp[i, j] *= 0

for i, j in ti.ndrange(N, N):
    x[i, j] = ti.Vector(
        [(i + 0.5) * L - 0.5 * W, (j + 0.5) * L / ti.sqrt(2) + 1.0, (N - j) * L / ti.sqrt(2) - 0.4 * W]
    )

    if i < N - 1 and j < N - 1:
        tri_id = ((i * (N - 1)) + j) * 2
        indices[tri_id].x = i * N + j
        indices[tri_id].y = (i + 1) * N + j
        indices[tri_id].z = i * N + (j + 1)

        tri_id += 1
        indices[tri_id].x = (i + 1) * N + j + 1
        indices[tri_id].y = i * N + (j + 1)
        indices[tri_id].z = (i + 1) * N + j

@ti.func
def flat2index(flat):
    j = flat % N
    i = (flat - j) / N
    return i, j

@ti.func
def calc_angle(p1, p2, p3, p4):
    x1 = x[flat2index(p1)]
    x2 = x[flat2index(p2)]
    x3 = x[flat2index(p3)]
    x4 = x[flat2index(p4)]
    # do not consider x1
    n1 = ti.cross(x2, x3) / ti.cross(x2, x3).norm()
    n2 = ti.cross(x2, x4) / ti.cross(x2, x4).norm()
    # TODO clamp the n1*n2 in the range of (-1, 1)
    return ti.acos(ti.dot(n1, n2))


@ti.func
class BendingConstraints:
    def __init__(self, stiffness):
        # store constraint, each bend_indices has a item of (p1, p2, p3, p4, constrain angle)
        self.bend_indices = ti.Vector.field(5, float, 3 * num_triangles)
        for i in range(3 * num_triangles):
            self.bend_indices[i] *= 0
        self.build_constrain()
        self.stiffness = stiffness

    def build_constrain(self):
        for i in range(num_triangles):
            if i % 2 == 1:
                p1 = indices[i].x
                p2 = indices[i].y
                p3 = indices[i].z
                p4_1 = p2 + 1
                p4_2 = p2 - 1
                p4_3 = p3 + N
                x1, x2 = flat2index(p4_1)
                if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
                    constrain_id = i * 3
                    bend_indices[constrain_id][0] = p1
                    bend_indices[constrain_id][1] = p2
                    bend_indices[constrain_id][2] = p3
                    bend_indices[constrain_id][3] = p4_1
                    bend_indices[constrain_id][4] = calc_angle(p1, p2, p3, p4_1)
                x1, x2 = flat2index(p4_3)
                if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
                    # (p1, p3, p2, p4_3)
                    constrain_id = i * 3 + 1
                    bend_indices[constrain_id][0] = p1
                    bend_indices[constrain_id][1] = p3
                    bend_indices[constrain_id][2] = p2
                    bend_indices[constrain_id][3] = p4_3
                    bend_indices[constrain_id][4] = calc_angle(p1, p3, p2, p4_3)
                x1, x2 = flat2index(p4_2)
                if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
                    # (p2, p3, p1, p4_2)
                    constrain_id = i * 3 + 2
                    bend_indices[constrain_id][0] = p2
                    bend_indices[constrain_id][1] = p3
                    bend_indices[constrain_id][2] = p1
                    bend_indices[constrain_id][3] = p4_2
                    bend_indices[constrain_id][4] = calc_angle(p2, p3, p1, p4_2)

            if i % 2 == 0:
                p1 = indices[i].x
                p2 = indices[i].y
                p3 = indices[i].z
                p4_1 = p2 + 1
                p4_2 = p2 - 1
                p4_3 = p3 - N
                x1, x2 = flat2index(p4_1)
                if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
                    constrain_id = i * 3
                    bend_indices[constrain_id][0] = p2
                    bend_indices[constrain_id][1] = p3
                    bend_indices[constrain_id][2] = p1
                    bend_indices[constrain_id][3] = p4_1
                    bend_indices[constrain_id][4] = calc_angle(p2, p3, p1, p4_1)
                x1, x2 = flat2index(p4_2)
                if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
                    constrain_id = i * 3 + 1
                    bend_indices[constrain_id][0] = p1
                    bend_indices[constrain_id][1] = p2
                    bend_indices[constrain_id][2] = p3
                    bend_indices[constrain_id][3] = p4_2
                    bend_indices[constrain_id][4] = calc_angle(p1, p2, p3, p4_2)
                x1, x2 = flat2index(p4_3)
                if x1 < N - 1 and x1 > 0 and x2 < N - 1 and x2 > 0:
                    # (p2, p3, p1, p4_2)
                    constrain_id = i * 3 + 2
                    bend_indices[constrain_id][0] = p1
                    bend_indices[constrain_id][1] = p3
                    bend_indices[constrain_id][2] = p2
                    bend_indices[constrain_id][3] = p4_3
                    bend_indices[constrain_id][4] = calc_angle(p1, p3, p2, p4_3)
        # for c in bend_indices:
        #     print(bend_indices[c])


    def project(self):
        for i in range(3 * num_triangles):
            p1_index, p2_index, p3_index, p4_index, constrain_angle = self.bend_indices[i]
            p1, p2, p3, p4 = x[p1_index], x[p2_index], x[p3_index], x[p4_index]
            p2Xp3 = ti.cross(p2,p3)
            p2Xp4 = ti.cross(p2,p4)
            n1 = p2Xp3 / p2Xp3.norm()
            n2 = p2Xp4 / p2Xp4.norm()
            d = ti.dot(n1,n2)
            q3 = (p2.cross(n2) + d * n1.cross(p2)) / (p2Xp3.norm())
            q4 = (p2.cross(n1) + d * n2.cross(p2)) / (p2Xp4.norm())
            q2 = -(p3.cross(n2) + d * n1.cross(p3)) / (p2Xp3.norm()) - (p4.cross(n1) + d * n2.cross(p4)) / (
                p2Xp4.norm())
            q1 = -q2 - q3 - q4
            # assume all the point has same mass
            # TODO clamp the d in the range of (-1, 1)
            a = ti.sqrt(1.0 - d * d) * (ti.acos(d) - constrain_angle)
            qSum = q1.norm_sqr() + q2.norm_sqr() + q3.norm_sqr() + q4.norm_sqr()
            displacements_p1 = (-a * q1) / qSum
            displacements_p2 = (-a * q2) / qSum
            displacements_p3 = (-a * q3) / qSum
            displacements_p4 = (-a * q4) / qSum
            # update replacement
            x[p1_index] += displacements_p1 * self.stiffness
            x[p2_index] += displacements_p2 * self.stiffness
            x[p3_index] += displacements_p3 * self.stiffness
            x[p4_index] += displacements_p4 * self.stiffness