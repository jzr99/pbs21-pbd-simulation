# import numpy as np
import taichi as ti
# import open3d as o3d
# import utils
from functools import partial


class BendingConstraints:
    def __init__(self, mesh, bend_factor, solver_iterations):
        # store constraint, each bend_indices has a item of (p1, p2, p3, p4, constrain angle)
        self.bend_indices = ti.Vector.field(5, float, len(mesh._adjacent_triangles))
        self.bend_mask = ti.Vector.field(1, ti.u32, len(mesh._adjacent_triangles))
        self.mesh = mesh
        self.adjacent_triangles = mesh._adjacent_triangles
        self.build_constrain()
        self.stiffness = 1.0 - pow(1.0 - bend_factor, 1.0 / solver_iterations)

    # @ti.func
    def calc_angle(self, p1, p2, p3, p4):
        EPSILON = 1e-8
        x1 = self.mesh.vertices[p1]
        x2 = self.mesh.vertices[p2]
        x3 = self.mesh.vertices[p3]
        x4 = self.mesh.vertices[p4]
        n1 = (x2-x1).cross(x3-x1) / (x2-x1).cross(x3-x1).norm()
        n2 = (x2-x1).cross(x4-x1) / (x2-x1).cross(x4-x1).norm()
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

    # file format p1_index p2_index p3_index p4_index angle
    @ti.pyfunc
    def write_angle_csv(self, file_name='./bending_angle.csv'):
        import csv
        EPSILON = 1e-8
        with open(file_name, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in ti.ndrange(*self.bend_mask.shape):
                # TODO filter zero constrain
                if self.bend_mask[i][0] == 1:
                    # print('bend_indices', self.bend_indices[i])
                    p1_index, p2_index, p3_index, p4_index, constrain_angle = self.bend_indices[i]
                    p1, p2, p3, p4 = self.mesh.estimated_vertices[int(p1_index)], self.mesh.estimated_vertices[int(p2_index)], \
                                     self.mesh.estimated_vertices[int(p3_index)], self.mesh.estimated_vertices[int(p4_index)]
                    p2_new = p2 - p1
                    p3_new = p3 - p1
                    p4_new = p4 - p1
                    p2Xp3 = p2_new.cross(p3_new)
                    p2Xp4 = p2_new.cross(p4_new)
                    n1 = p2Xp3 / (p2Xp3.norm() + EPSILON)
                    n2 = p2Xp4 / (p2Xp4.norm() + EPSILON)
                    d = n1.dot(n2)
                    angle = ti.acos(d)
                    writer.writerow([int(p1_index), int(p2_index), int(p3_index), int(p4_index), float(angle)])


    @ti.pyfunc
    def project(self):
        EPSILON = 1e-8
        # for i in ti.grouped(self.bend_mask):
        for i in ti.ndrange(self.bend_mask.shape[0]):
            # TODO filter zero constrain
            if self.bend_mask[i][0] == 1:
                p1_index, p2_index, p3_index, p4_index, constrain_angle = self.bend_indices[i]
                p1, p2, p3, p4 = self.mesh.estimated_vertices[int(p1_index)], self.mesh.estimated_vertices[int(p2_index)], self.mesh.estimated_vertices[int(p3_index)], self.mesh.estimated_vertices[int(p4_index)]
                p2_new = p2 - p1
                p3_new = p3 - p1
                p4_new = p4 - p1
                p2Xp3 = p2_new.cross(p3_new)
                p2Xp4 = p2_new.cross(p4_new)
                n1 = p2Xp3 / (p2Xp3.norm() + EPSILON)
                n2 = p2Xp4 / (p2Xp4.norm() + EPSILON)
                d = n1.dot(n2)
                d = max(min(d, 1.0 - EPSILON), -1.0 + EPSILON)
                q3 = (p2_new.cross(n2) + d * n1.cross(p2_new)) / (p2Xp3.norm()+EPSILON)
                q4 = (p2_new.cross(n1) + d * n2.cross(p2_new)) / (p2Xp4.norm()+EPSILON)
                q2 = -(p3_new.cross(n2) + d * n1.cross(p3_new)) / (p2Xp3.norm()+EPSILON) - (p4_new.cross(n1) + d * n2.cross(p4_new)) / (
                    p2Xp4.norm()+EPSILON)
                q1 = -q2 - q3 - q4
                # assume all the point has same mass
                a = ti.sqrt(1.0 - d * d) * (ti.acos(d) - constrain_angle)
                qSum = q1.norm_sqr() + q2.norm_sqr() + q3.norm_sqr() + q4.norm_sqr()
                displacements_p1 = (-a * q1) / qSum
                displacements_p2 = (-a * q2) / qSum
                displacements_p3 = (-a * q3) / qSum
                displacements_p4 = (-a * q4) / qSum
                self.mesh.estimated_vertices[p1_index] += displacements_p1 * self.stiffness
                self.mesh.estimated_vertices[p2_index] += displacements_p2 * self.stiffness
                self.mesh.estimated_vertices[p3_index] += displacements_p3 * self.stiffness
                self.mesh.estimated_vertices[p4_index] += displacements_p4 * self.stiffness