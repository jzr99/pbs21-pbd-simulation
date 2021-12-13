import taichi as ti


@ti.data_oriented
class CollisionConstraints:
    def __init__(self, dynamic_meshes):
        self.dynamic_meshes = dynamic_meshes
        num_dynamic_ver = []
        for mesh in dynamic_meshes:
            num_dynamic_ver.append(mesh.num_vertices)

        self.mesh_index = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))  # deprecated
        self.vertice_index = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))  # deprecated

        cnt = 0
        for mesh_idx, mesh_num in enumerate(num_dynamic_ver):
            for i in range(mesh_num):
                cnt += 1
                self.mesh_index[cnt] = mesh_idx
                self.vertice_index[cnt] = i

        self.has_constraint = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))
        self.surface_norm = ti.Vector.field(3, dtype=ti.float32, shape=(sum(num_dynamic_ver)))
        self.entry_point = ti.Vector.field(3, dtype=ti.float32, shape=(sum(num_dynamic_ver)))

        # dynamic collision global data structure
        self.has_self_constraint = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))
        self.self_collision_surface_norm = ti.Vector.field(3, dtype=ti.float32, shape=(sum(num_dynamic_ver)))
        self.self_collision_entry_point = ti.Vector.field(3, dtype=ti.float32, shape=(sum(num_dynamic_ver)))
        self.self_other_vertices_idx = ti.Vector.field(3, dtype=ti.int32, shape=(sum(num_dynamic_ver)))
        self.self_collision_sum = ti.field(dtype=ti.float32, shape=(), needs_grad=True)

    @ti.kernel
    def reset(self):
        for i in ti.grouped(self.has_constraint):
            self.has_constraint[i] = 0

            self.has_self_constraint[i] = 0

            self.self_collision_sum[None] = 0

            self.surface_norm[i].x = 0
            self.surface_norm[i].y = 0
            self.surface_norm[i].z = 0

            self.entry_point[i].x = 0
            self.entry_point[i].y = 0
            self.entry_point[i].z = 0

            self.self_other_vertices_idx[i].x = 0
            self.self_other_vertices_idx[i].y = 0
            self.self_other_vertices_idx[i].z = 0

            self.self_collision_surface_norm[i].x = 0
            self.self_collision_surface_norm[i].y = 0
            self.self_collision_surface_norm[i].z = 0

            self.self_collision_entry_point[i].x = 0
            self.self_collision_entry_point[i].y = 0
            self.self_collision_entry_point[i].z = 0

    @ti.func
    def add_constraint(self, global_index, norm, entry_point):
        if self.has_self_constraint[global_index] == 1:
            print('warning: constraint should be added only once. ')
        self.has_constraint[global_index] = 1
        self.surface_norm[global_index] = norm
        self.entry_point[global_index] = entry_point

    @ti.func
    def add_self_constraint(self, global_index, p1, p2, p3, norm, entry_point):
        self.has_self_constraint[global_index] = 1
        self.self_collision_surface_norm[global_index] = norm
        self.self_collision_entry_point[global_index] = entry_point
        self.self_other_vertices_idx[global_index].x = p1
        self.self_other_vertices_idx[global_index].y = p2
        self.self_other_vertices_idx[global_index].z = p3

    @ti.func
    def project(self, global_var_idx, p: ti.template()):

        if self.has_constraint[global_var_idx] == 0:
            pass
        else:
            # apply project
            entry_to_p = p - self.entry_point[global_var_idx]
            surface_norm = self.surface_norm[global_var_idx]
            # if dot product is negative, the collision constraint already been solved.
            if entry_to_p.dot(surface_norm) >= 0:
                pass
            else:
                disp_length = -entry_to_p.dot(surface_norm)  # new
                # disp_length = -entry_to_p.norm()  # todo how much offset do we need push back ?
                p = p + disp_length * surface_norm  # new
                # p = p + disp_length * entry_to_p.normalized()

    @ti.func
    def calibrate_colliding_vertices(self, global_var_idx: int, v: ti.template()):
        if self.has_constraint[global_var_idx] == 0:  # todo consider self collision ?
            pass

        else:
            # get the mesh index and vertice index of the constrained vertices

            # apply velocity reflection
            surface_norm = self.surface_norm[global_var_idx]
            v = v - 2 * v.dot(surface_norm) * surface_norm

            v = v.dot(surface_norm) * 0.01 * surface_norm + (v - v.dot(surface_norm) * surface_norm) * (1 - 0.1)

            # todo if friction and restitution exists


@ti.func
def inner_angle_check(o, other1, other2, p):
    return_flag = 1
    edge1_norm = (other1 - o).normalized()
    edge2_norm = (other2 - o).normalized()
    op_norm = (p - o).normalized()
    if op_norm.dot(edge1_norm) < edge2_norm.dot(edge1_norm):
        return_flag = 0

    return return_flag


@ti.func
def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2):
    """
    this function only makes sure than the ray can collide the triangle along the specified direction.
    since it does not the length of ray, further comparison between ray length and distance to the entry point (t) is needed.
    :param ray_origin: original location of collision vertices before the simulation at this epoch.
    :param ray_direction: normalized moving direction.
    :param v0
    :param v1
    :param v2
    :return: t > 0 or -1
    """
    # ---------- Preprocess Begin ----------
    # if triangle_index == ...:
    #     return False
    # v0 = ti.Vector([0, 0, 0])
    # v1 = ti.Vector([0, 0, 0])
    # v2 = ti.Vector([0, 0, 0])
    # ---------- Preprocess End ----------

    return_flag = ti.Vector([1.0], dt=ti.float32)

    # Special Case1: ray parallel to triangle
    norm = (v1 - v0).cross(v2 - v0)
    if abs(norm.dot(ray_direction)) <= 0.0001:
        return_flag[0] = -1

    else:
        # Compute t
        d = - norm.dot(v0)  # ax + by + cz + d = 0, (a, b, c) derived from norm vector
        t = - (norm.dot(ray_origin) + d) / norm.dot(ray_direction)

        # Special Case2: triangle behind the ray direction
        if t < 0:
            return_flag[0] = -1

        # Special Case3: entry point out of the triangle area
        entry_point = ray_origin + ray_direction * t

        if not inner_angle_check(v0, v1, v2, entry_point):
            return_flag[0] = -1
        if not inner_angle_check(v0, v2, v1, entry_point):
            return_flag[0] = -1
        if not inner_angle_check(v1, v0, v2, entry_point):
            return_flag[0] = -1
        if not inner_angle_check(v1, v2, v0, entry_point):
            return_flag[0] = -1
        if not inner_angle_check(v2, v0, v1, entry_point):
            return_flag[0] = -1
        if not inner_angle_check(v2, v1, v0, entry_point):
            return_flag[0] = -1

        if return_flag[0] == 1.0:
            return_flag[0] = float(t)

    return return_flag
