import taichi as ti
import sys
import time
import numpy as np
from simulator import Simulation
from collections import defaultdict

from taichi.misc.util import vec

from bounding_box import BoundingBox
import utils


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
    def __init__(self, filename, color, inverse_mass=1.0, rescale=1.0, translation=[0, 0, 0]) -> None:
        self.color = color
        self.inverse_mass = inverse_mass
        self.rescale = rescale
        self.translation = translation

        self.parse_file(filename)  # read obj mesh file

        self.initial_vertices = ti.Vector.field(3, ti.f32, self.num_vertives)
        self.initial_vertices.copy_from(self.vertices)

        self.position = ti.field(dtype=ti.f32, shape=(3))  # position of COM
        self.velocities = ti.Vector.field(3, ti.f32, (self.num_vertives))

        self.bounding_box = BoundingBox()
        self.bounding_box.update_bounding_box(self.vertices)

        self.gravity_affected = False
        self.wind_affected = False

    def parse_file(self, filename):
        vertices = []
        uvs = []
        normals = []
        edges = []
        triangles = []
        adjacent_triangles = defaultdict(list)

        with open(filename, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(" ")
                if items[0] == 'v':  # vertex
                    vertices.append(np.array(items[1:], dtype=np.float32))
                elif items[0] == 'vn':  # vertex normal
                    normals.append(np.array(items[1:], dtype=np.float32))
                elif items[0] == 'vt':  # UV
                    uvs.append(np.array(items[1:], dtype=np.float32))
                elif items[0] == 'f':
                    verts = []
                    for item in items[1:]:
                        v_items = item.split('/')
                        verts.append(Vertex(int(v_items[0]), int(v_items[1]), int(v_items[2])))
                    if len(verts) == 3:
                        triangles.append(verts)
                        e1 = Edge(verts[0], verts[1])
                        e2 = Edge(verts[0], verts[2])
                        e3 = Edge(verts[1], verts[2])
                        edges.extend([e1, e2, e3])
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
        self.num_vertives = len(vertices)
        self.num_face = len(triangles)
        self._num_uvs = len(uvs)  # not used
        self.edges = edges
        self._adjacent_triangles = adjacent_triangles  # not used

        self.triangle = triangles

        self.vertices = ti.Vector.field(3, ti.float32, self.num_vertives)
        vertices = np.array(vertices)
        vertices[..., 0] = vertices[..., 0] - np.mean(x)
        vertices[..., 1] = vertices[..., 1] - np.mean(y)
        vertices[..., 2] = vertices[..., 2] - np.mean(z)
        vertices = vertices * self.rescale + np.array(self.translation).reshape(-1, 3)
        self.vertices.from_numpy(np.array(vertices))
        self.vertices = self.vertices

        self.estimated_vertices = ti.Vector.field(3, ti.float32, self.num_vertives)
        self.estimated_vertices.from_numpy(np.array(vertices))

        self._normals = ti.Vector.field(3, ti.float32, self.num_face)  # not used
        self._normals.from_numpy(np.array(normals))

        self._uvs = ti.Vector.field(3, ti.float32, self._num_uvs)  # not used
        self._uvs.from_numpy(np.array(uvs) * self.rescale)

        self._surface_normals = ti.Vector.field(3, ti.f32, (self.num_face))  # not used
        self.generate_surface_normals()

        self.indices = ti.Vector.field(3, ti.f32, (self.num_face))
        self.generate_triangle_indices()

    def generate_surface_normals(self):
        for i in range(self.num_face):
            vector1 = self.vertices[self.triangle[i][1].p] - self.vertices[self.triangle[i][0].p]
            vector2 = self.vertices[self.triangle[i][2].p] - self.vertices[self.triangle[i][0].p]
            norm = vector1.cross(vector2)
            norm = norm.normalized()
            self._surface_normals[i] = norm

    def generate_triangle_indices(self):
        """
        Used for set triangles, each element contains three indexes, specifying the three corner vertices.
        """
        for i in range(self.indices.shape[0]):
            self.indices[i][0] = self.triangle[i][0].p
            self.indices[i][1] = self.triangle[i][1].p
            self.indices[i][2] = self.triangle[i][2].p

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
        for i in ti.ndrange(*self.velocities.shape):
            self.velocities[i] += force

    @ti.func
    def translate(self, translate):
        for i in ti.ndrange(*self.vertices.shape):
            self.vertices[i] += translate

    def intersect(self):
        raise NotImplementedError()

    def export_for_render(self):
        return self.vertices, self.indices, self.color



# test
from render.render import Render

if __name__ == '__main__':
    ti.init()

    # X, Y, Z 对应关系
    # X，Z：平面坐标，X控制水平方向，从左到右依次递增；Z控制竖直方向，从后到前依次递增。（原点在初始视角平面的左上角）
    # Y：纵向坐标
    # rescale：放大缩小使用；translation：尺度变换后，物体的平移
    mesh_sphere = Mesh(filename='../obj/sphere.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0, 0.6, 0])
    mesh_cloth = Mesh(filename='../obj/cloth.obj', color=[0.5, 0.5, 0.5], rescale=0.1, translation=[0, 1.0, 0])

    rendering_data = mesh_sphere.export_for_render()
    render = Render({'sphere': mesh_sphere.export_for_render(), 'cloth': mesh_cloth.export_for_render()})
    sim = Simulation(mesh_sphere, mesh_cloth, render)

    while True:
        sim.update()
        # this conditaional code is very important
        # if not render.vis.poll_events():
        #     break
        # render.vis.update_renderer()

