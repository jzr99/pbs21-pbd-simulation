import taichi as ti


@ti.data_oriented
class CollisionConstraints:
    def __init__(self, dynamic_meshes):
        num_dynamic_ver = []
        for mesh in dynamic_meshes:
            num_dynamic_ver.append(mesh.num_vertices)

        self.mesh_index = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))
        self.vertice_index = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))

        cnt = 0
        for mesh_idx, mesh_num in enumerate(num_dynamic_ver):
            for i in range(mesh_num):
                cnt += 1
                self.mesh_index[cnt] = mesh_idx
                self.vertice_index[cnt] = i

        self.has_constraint = ti.field(dtype=ti.int32, shape=(sum(num_dynamic_ver)))
        self.surface_norm = ti.Vector.field(3, dtype=ti.float32, shape=(sum(num_dynamic_ver)))
        self.entry_point = ti.Vector.field(3, dtype=ti.float32, shape=(sum(num_dynamic_ver)))

    @ti.kernel
    def reset(self):
        for i in ti.grouped(self.has_constraint):
            self.has_constraint[i] = 0

            self.surface_norm[i].x = 0
            self.surface_norm[i].y = 0
            self.surface_norm[i].z = 0

            self.entry_point[i].x = 0
            self.entry_point[i].y = 0
            self.entry_point[i].z = 0

    @ti.func
    def add_constraint(self):
        pass

    @ti.func
    def project(self, dynamic_meshes):
        pass


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
        if not inner_angle_check(v1, v0, v2, entry_point):
            return_flag[0] = -1
        if not inner_angle_check(v2, v0, v1, entry_point):
            return_flag[0] = -1

        if return_flag[0] == 1.0:
            return_flag[0] = float(t)

    return return_flag


class CollisionConstraints_Wrong:

    def __init__(self, index: int, normal: ti.Vector, entry_point: ti.Vector):
        """
        :param index: index of projected point in global data. (e.g. configuration.EstimatedPosition)
        :param normal: normal position of triangle (normal must in opposite direction against ray-direction)
        :param entry_point: position of entry point on the triangle
        """
        self.index = index
        self.normal = normal
        self.entry_point = entry_point

    def project(self, configuration, **kwargs):
        """
        :param configuration: global structure to store temporal data (e.g. configuration.estimateposition for un-projected data).
        """
        # ----- pre-process -----
        # todo get current estimate position
        p = ti.Vector([0, 0, 0])
        # -----------------------
        p_to_entry = p - self.entry_point
        p_to_entry_norm = p_to_entry.norm()

        # since projection solver run multiple times, this constraint may be solved already.
        if ti.dot(p_to_entry, self.normal) >= 0:
            return

        # displacement along the direction of entering point. (Hint: in paper, the direction should be along with normal vector).
        disp_length = ti.dot(p_to_entry, self.normal)
        disp = disp_length * p_to_entry_norm

        # ----- post-process -----
        # todo update position in configuration data
        # configuration.estimate_position[self.index] += disp
        # ------------------------
