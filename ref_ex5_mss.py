import sys
import numpy as np
import taichi as ti
import open3d as o3d
from functools import partial

ti.init(arch=ti.cpu)

### Parameters

N = 64 #128
W = 2
L = W / N
gravity = 9.8
mass = 0.1
stiffness = 600
object_stiffness = 10000
damping = 1.5
steps = 15
dt = 1e-3
paused = False

num_balls = 1
ball_radius = 0.3
ball_centers = ti.Vector.field(3, float, num_balls)

x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))
fint = ti.Vector.field(3, float, (N, N))
fext = ti.Vector.field(3, float, (N, N))
fdamp = ti.Vector.field(3, float, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.Vector.field(3, int, num_triangles)
vertices = ti.Vector.field(3, float, N * N)

links = ti.Vector.field(2, int, 8)
links.from_numpy(np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]]))
links_end = 8 # take the first 8 elements of links


### EX 5 - Simulation

@ti.kernel
def init():
    for i, j in ti.ndrange(N, N):
        x[i, j] *= 0
        v[i, j] *= 0
        fint[i, j] *= 0
        fext[i, j] *= 0
        fdamp[i, j] *= 0

    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector(
            [(i + 0.5) * L - 0.5 * W, (0.5 + 0.5) * L / ti.sqrt(2) + 1.0, j * L / ti.sqrt(2) - 0.4 * W]
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
    ball_centers[0] = ti.Vector([0.0, 0.5, 0.0])


@ti.func
def objectBoundImpulse(pos, vel, center, radius, bounce_magnitude=0.1):
    ret = vel * 0
    # impulse wrt ball
    distance = (pos - center).norm()
    if distance <= radius:
        normal = (pos - center).normalized()
        ret =  -normal * object_stiffness * (distance - radius)
    # impulse wrt ground plane
    if pos.y < 0:
        ret.y = -object_stiffness * pos.y
    return ret


@ti.kernel
def substep(links_end: ti.i32):
    for i in ti.grouped(x):
        # internal forces
        fint[i] = x[i] * 0
        for k in range(links_end):
            d = links[k]
            disp = x[min(max(i + d, 0), ti.Vector([N - 1, N - 1]))] - x[i]
            length = L * float(d).norm()
            if disp.norm() > 0:
                fint[i] +=  disp * (stiffness / length) * (disp.norm() - length) / disp.norm()

        # external forces (gravity and elastic impulse)
        fext[i] = ti.Vector([0.0, -mass * gravity, 0.0])
        for b in range(num_balls):
            fext[i] += objectBoundImpulse(x[i], v[i], ball_centers[b], ball_radius * 1.01)

        # damping forces
        fdamp[i] = damping * v[i]

    # semi-implicit Euler update
    for i in ti.grouped(x):
        v[i] += (fext[i] + fint[i] - fdamp[i]) * dt / mass
        x[i] += dt * v[i]


@ti.kernel
def update_verts():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]


### GUI

# key callback functions
def reset_sim(vis):
    init()
    update_verts()

def pause_sim(vis):
    global paused
    paused = not paused

def set_mss(vis, type):
    global links_end
    if type == 1:
        links_end = 8
    elif type == 2:
        links_end = 4
    elif type == 3:
        links_end = 6


# init and setup gui

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
reset_sim(vis)

# set callbacks
vis.register_key_callback(ord("R"), reset_sim)
vis.register_key_callback(ord(" "), pause_sim)  # space
vis.register_key_callback(ord("1"), partial(set_mss, type=1)) # crossed
vis.register_key_callback(ord("2"), partial(set_mss, type=2)) # opposite
vis.register_key_callback(ord("3"), partial(set_mss, type=3)) # regular

# MSS cloth mesh
V = o3d.utility.Vector3dVector(vertices.to_numpy())
F = o3d.utility.Vector3iVector(indices.to_numpy())
mesh = o3d.geometry.TriangleMesh(V, F)
mesh.paint_uniform_color([0.5, 0.5, 0.5])
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
vis.add_geometry(mesh)

# sphere mesh
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=ball_radius, resolution=60)
sphere.translate(ball_centers[0])
sphere.paint_uniform_color([1.0, 0.4, 0.2])
sphere.compute_vertex_normals()
vis.add_geometry(sphere)

# stanford bunny
bunny = o3d.io.read_triangle_mesh('obj/bun_zipper_res4.ply')
bunny.paint_uniform_color([1.0, 0.4, 0.2])
bunny.compute_vertex_normals()
bunny.compute_triangle_normals()
vis.add_geometry(bunny)

# ground plane mesh
def create_ground_plane():
    N = 16
    vertices = np.zeros((N * N, 3))
    indices = np.zeros(((N-1) * (N-1) * 2, 3))
    for i in range(N):
        for j in range(N):
            vertices[i * N + j] = np.array([(i / (N-1) - 0.5) * 2, 0.0, (j / (N-1) - 0.5) * 2])  # [-1, 1]; [0]; [-1, 1]

            if i < N - 1 and j < N - 1:
                tri_id = ((i * (N - 1)) + j) * 2
                indices[tri_id][0] = i * N + j
                indices[tri_id][1] = (i + 1) * N + j
                indices[tri_id][2] = i * N + (j + 1)

                tri_id += 1
                indices[tri_id][0] = (i + 1) * N + j + 1
                indices[tri_id][1] = i * N + (j + 1)
                indices[tri_id][2] = (i + 1) * N + j

    return vertices, indices

ground_vertice, ground_indices = create_ground_plane()
mesh_ground = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(ground_vertice), o3d.utility.Vector3iVector(ground_indices))
mesh_ground.paint_uniform_color([0.8, 0.8, 0.8])
mesh_ground.compute_vertex_normals()
mesh_ground.compute_triangle_normals()
vis.add_geometry(mesh_ground)

# plane mesh
points = (
        [[i/10, 0, -1] for i in range(-10, 11)]
        + [[i/10, 0, 1] for i in range(-10, 11)]
        + [[-1, 0, i/10] for i in range(-10, 11)]
        + [[1, 0, i/10] for i in range(-10, 11)]
    )
lines = [[i, i + 21] for i in range(21)] + [[i + 42, i + 63] for i in range(21)]
colors = [[0.7, 0.7, 0.7] for i in range(len(lines))]
ground_plane = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
ground_plane.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(ground_plane)

# render and view options
rdr = vis.get_render_option()
rdr.mesh_show_back_face = True
# rdr.mesh_show_wireframe = True
ctr = vis.get_view_control()
ctr.set_lookat([0.0, 0.5, 0.0])
ctr.set_up([0.0, 1.0, 0.0])

while True:
    update_verts()

    if not paused:
        for i in range(steps):
            substep(links_end)

    mesh.vertices = o3d.utility.Vector3dVector(vertices.to_numpy())
    mesh.triangles = o3d.utility.Vector3iVector(indices.to_numpy())
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    vis.update_geometry(mesh)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()
    mesh_ground.compute_vertex_normals()
    mesh_ground.compute_triangle_normals()
    bunny.compute_vertex_normals()
    bunny.compute_triangle_normals()
    o3d.io.write_triangle_mesh('obj/cloth_large_new.obj', mesh=mesh)
    o3d.io.write_triangle_mesh('obj/sphere_large_new.obj', mesh=sphere)
    o3d.io.write_triangle_mesh('obj/ground.obj', mesh=mesh_ground)
    o3d.io.write_triangle_mesh('obj/bunny.obj', mesh=bunny)
    break

    if not vis.poll_events():
        break
    vis.update_renderer()
