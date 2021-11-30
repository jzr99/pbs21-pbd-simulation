from _typeshed import Self
from os import O_EXCL
import taichi as ti
import sys
import numpy as np
from collections import defaultdict

from taichi.misc.util import vec

from bounding_box import BoudingBox


@ti.func
def clear_field(f: ti.template(), v: ti.template() = 0):
    for x, y in ti.ndrange(*f.shape):
        f[x, y] = v


class Vertex:
    def __init__(self, p: int, t: int, n: int) -> None:
        self.p = p - 1
        self.t = t - 1
        self.n = n - 1

    def __eq__(self, other: object) -> bool:
        return self.p == other.p and self.t == other.t and self.n == other.n

    def __hash__(self) -> int:
        return hash((self.p, self.t, self.n))


class Edge:
    def __init__(self, v1: Vertex, v2: Vertex) -> None:
        self.v1 = v1
        self.v2 = v2

    def __eq__(self, other) -> bool:
        return self.v1 == other.v1 and self.v2 == other.v2

    def __hash__(self) -> int:
        return hash((self.v1, self.v2))


class Mesh:
    def __init__(self, filename, color, inverse_mass=1.0) -> None:
        self.color = color
        self.inverse_mass = inverse_mass

        self.parse_file(filename)  # read obj mesh file

        self.initial_vertices = self.vertices
        self.postion = ti.field(dtype=ti.f32, shape=(3))
        self.velocities = ti.Vector.field(3, ti.f32, (self.num_vertives))

        self.bounding_box = BoudingBox()
        self.update_bounding_box()

    def parse_file(self, filename):
        self.vertices = []
        self.normals = []
        self.uvs = []
        self.edges = []
        self.triangles = []
        self.adjacent_triangles = defaultdict(list)

        with open(filename, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(" ")
                if items[0] == 'v':  # vertex
                    self.vertices.append(np.array(items[1:]))
                elif items[0] == 'vn':  # vertex normal
                    self.normals.append(np.array(items[1:]))
                elif items[0] == 'vt':  # UV
                    self.uvs.append(np.array(items[1:]))
                elif items[0] == 'f':
                    verts = []
                    for item in items[1:]:
                        v_items = item.split('/')
                        verts.append(Vertex(v_items[0], v_items[1], v_items[2]))
                    if len(verts) == 3:
                        self.triangles.append(verts)
                        e1 = Edge(verts[0], verts[1])
                        e2 = Edge(verts[0], verts[2])
                        e3 = Edge(verts[1], verts[2])
                        self.edges.extend([e1, e2, e3])
                        self.adjacent_triangles[e1].append(verts)
                        self.adjacent_triangles[e2].append(verts)
                        self.adjacent_triangles[e3].append(verts)
        self.num_vertives = len(self.vertices)
        self.num_face = len(self.triangles)
        self.vertices = np.array(self.vertices)
        self.normals = np.array(self.normals)
        self.uvs = np.array(self.uvs)
        self.generate_surface_normals()

    def generate_surface_normals(self):
        tri_ver0 = self.vertices[self.triangles[:, 0]]
        tri_ver1 = self.vertices[self.triangles[:, 1]]
        tri_ver2 = self.vertices[self.triangles[:, 2]]
        vector1 = tri_ver1 - tri_ver0
        vector2 = tri_ver2 - tri_ver0
        self.surface_normals = np.cross(vector1, vector2)
        self.surface_normals = self.surface_normals / np.linalg.norm(self.surface_normals, axis=1)

    def reset(self):
        self.vertices = self.initial_vertices
        clear_field(self.velocities)

    # void Mesh::applyImpulse(Vector3f force) {
    #     for (int i = 0; i < numVertices; i++) {
    #         velocities[i] += force;
    #     }
    # }

    # void Mesh::translate(Vector3f translate) {
    #     for (int i = 0; i < numVertices; i++) {
    #         vertices[i] += translate;
    #     }
    # }

    def apply_impulse(self, force):
        for i in ti.ndrange(*self.velocities.shape):
            self.velocities[i] += force

    def translate(self, translate):
        for i in ti.ndrange(*self.vertices.shape):
            self.vertices[i] += translate

    def intersect(self):
        pass

    def update_bounding_box(self):
        min_values = self.vertices.min(axis=0)
        max_values = self.vertices.max(axis=0)
        self.bounding_box.x_min = min_values[0]
        self.bounding_box.y_min = min_values[1]
        self.bounding_box.z_min = min_values[2]
        self.bounding_box.x_max = max_values[0]
        self.bounding_box.y_max = max_values[1]
        self.bounding_box.z_max = max_values[2]

    def render(self):
        pass
