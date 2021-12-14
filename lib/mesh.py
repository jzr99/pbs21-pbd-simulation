import taichi as ti
import sys
import time
import numpy as np
from collections import defaultdict
import open3d as o3d

from .bounding_box import BoundingBox
import lib.utils as utils


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
        return self.p == other.p

    def __hash__(self) -> int:
        return hash(self.p)


class Edge:
    def __init__(self, v1: Vertex, v2: Vertex) -> None:
        self.v1 = v1
        self.v2 = v2

    def __eq__(self, other) -> bool:
        return (self.v1 == other.v1 and self.v2 == other.v2) or (self.v1 == other.v2 and self.v2 == other.v1)

    def __hash__(self) -> int:
        return hash(self.v1.p + self.v2.p)


class Mesh:
    def __init__(self, filename, color, inverse_mass=1.0, rescale=1.0, translation=[0, 0, 0], reverse_triangle_verts=False) -> None:
        self.color = color
        self.inverse_mass = inverse_mass
        self.rescale = rescale
        self.translation = translation
        self.reverse_triangle_verts = reverse_triangle_verts

        self.parse_file(filename)  # read obj mesh file

        self.initial_vertices = ti.Vector.field(3, ti.f32, self.num_vertices)
        self.initial_vertices.copy_from(self.vertices)

        self.position = ti.field(dtype=ti.f32, shape=(3))  # position of COM
        self.velocities = ti.Vector.field(3, ti.f32, (self.num_vertices))

        self.bounding_box = BoundingBox()
        self.bounding_box.update_bounding_box(self.vertices)

        self.gravity_affected = False
        self.wind_affected = False

    def parse_file(self, filename):
        vertices = []
        uvs = []
        normals = []
        edges = set()
        triangles = []
        adjacent_triangles = defaultdict(list)

        with open(filename, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(" ")
                if items[0] == 'v':  # vertex
                    vertices.append(np.array(items[1:4], dtype=np.float32))
                elif items[0] == 'vn':  # vertex normal
                    normals.append(np.array(items[1:4], dtype=np.float32))
                elif items[0] == 'vt':  # UV
                    uvs.append(np.array(items[1:4], dtype=np.float32))
                elif items[0] == 'f':
                    verts = []
                    for item in items[1:]:
                        v_items = item.split('/')
                        verts.append(Vertex(int(v_items[0]), 0, 0))
                    if self.reverse_triangle_verts:
                        verts.reverse()
                    if len(verts) == 3:
                        triangles.append(verts)
                        e1 = Edge(verts[0], verts[1])
                        e2 = Edge(verts[0], verts[2])
                        e3 = Edge(verts[1], verts[2])
                        edges.add(e1)
                        edges.add(e2)
                        edges.add(e3)
                        adjacent_triangles[e1].append(verts)
                        adjacent_triangles[e2].append(verts)
                        adjacent_triangles[e3].append(verts)



        # statistic
        x, y, z = [], [], []
        max_range = -float('inf')
        max_xyz = None
        for v in vertices:
            x.append(v[0])
            y.append(v[1])
            z.append(v[2])
            if np.linalg.norm(v) > max_range:
                max_range = np.linalg.norm(v)
                max_xyz = v[0], v[1], v[2]

        print(f'[Object Statistics Before Process]: x {np.mean(x)}, y {np.mean(y)}, z {np.mean(z)}, max_range {max_range} @ {max_xyz}')

        # save global variables
        self.num_vertices = len(vertices)
        self.num_face = len(triangles)
        print('num of ver: {}, num of triangle: {}'.format(self.num_vertices, self.num_face))
        self._num_uvs = len(uvs)  # not used
        self.edges = edges
        self._adjacent_triangles = adjacent_triangles  # not used

        self.triangle = ti.Vector.field(3, ti.int32, self.num_face)
        triangles_p = [[tri[0].p, tri[1].p, tri[2].p] for tri in triangles]  # self.triangle only saves vertices index, ignore normal and uv
        self.triangle.from_numpy(np.array(triangles_p))

        self.initial_vertices = ti.Vector.field(3, ti.float32, self.num_vertices)
        self.vertices = ti.Vector.field(3, ti.float32, self.num_vertices)
        vertices = np.array(vertices)
        vertices[..., 0] = vertices[..., 0] - np.mean(x)
        vertices[..., 1] = vertices[..., 1] - np.mean(y)
        vertices[..., 2] = vertices[..., 2] - np.mean(z)
        vertices = vertices * self.rescale + np.array(self.translation).reshape(-1, 3)
        self.initial_vertices.from_numpy(np.array(vertices))
        self.vertices.from_numpy(np.array(vertices))
        self.vertices = self.vertices

        self.estimated_vertices = ti.Vector.field(3, ti.float32, self.num_vertices, needs_grad=True)
        self.estimated_vertices.from_numpy(np.array(vertices))

        # self._normals = ti.Vector.field(3, ti.float32, self.num_face)  # not used
        # self._normals.from_numpy(np.array(normals))
        #
        # self._uvs = ti.Vector.field(3, ti.float32, self._num_uvs)  # not used
        # self._uvs.from_numpy(np.array(uvs) * self.rescale)
        #
        # self._surface_normals = ti.Vector.field(3, ti.f32, (self.num_face))  # not used
        # self.generate_surface_normals()

        self.indices = ti.Vector.field(3, ti.f32, (self.num_face))
        self.generate_triangle_indices()

    def generate_surface_normals(self):
        for i in range(self.num_face):
            vector1 = self.vertices[self.triangle[i][1]] - self.vertices[self.triangle[i][0]]
            vector2 = self.vertices[self.triangle[i][2]] - self.vertices[self.triangle[i][0]]
            norm = vector1.cross(vector2)
            norm = norm.normalized()
            self._surface_normals[i] = norm

    def generate_triangle_indices(self):
        """
        Used for set triangles, each element contains three indexes, specifying the three corner vertices.
        """
        self.indices = self.triangle
        # for i in range(self.indices.shape[0]):
        #     self.indices[i][0] = self.triangle[i][0]
        #     self.indices[i][1] = self.triangle[i][1]
        #     self.indices[i][2] = self.triangle[i][2]

    def reset(self):
        self.vertices = self.initial_vertices
        clear_field(self.velocities)

    def set_gravity_affected(self, sign: bool):
        self.gravity_affected = sign

    def set_wind_affected(self, sign: bool):
        self.wind_affected = sign

    @ti.func
    def reset_estimated_vertices(self):
        utils.copy(self.vertices, self.estimated_vertices)

    @ti.func
    def apply_impulse(self, force):
        for i in range(4):
            self.velocities[i] += force

    @ti.func
    def apply_impulse_wind(self, force):
        for i in ti.ndrange(*self.velocities.shape):
            self.velocities[i] += force * (ti.random() + 1)

    @ti.func
    def translate(self, translate):
        for i in ti.ndrange(*self.vertices.shape):
            self.vertices[i] += translate

    def intersect(self):
        raise NotImplementedError()

    def export_for_render(self):
        return self.vertices, self.indices, self.color
