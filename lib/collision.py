import taichi as ti


@ti.func
def ray_triangle_intersect(ray_origin: ti.Vector, ray_direction: ti.Vector, vertex_index: int, triangle_index: int, mesh):
    """

    :param ray_origin: original location of collision vertices before the simulation at this epoch.
    :param ray_direction: normalized moving direction.
    :param vertex_index: int, index of collision vertices in the mesh object.
    :param triangle_index: int, index of triangle in the mesh object.
    :param mesh: object class, require mesh.triangles and mesh.vertices to get index and position of triangle vertices.
    :return: bool, dict
    """
    # ---------- Preprocess Begin ----------
    # todo match v0, v1, v2 with positions of triangle vertices
    ...
    if triangle_index == ...:
        return False
    v0 = ti.Vector([0, 0, 0])
    v1 = ti.Vector([0, 0, 0])
    v2 = ti.Vector([0, 0, 0])
    # ---------- Preprocess End ----------

    # Special Case1: ray parallel to triangle
    norm = ti.cross(v1 - v0, v2 - v0)
    if ti.dot(norm, ray_direction) <= 1e-4:
        return False, None

    # Compute t
    d = - ti.dot(norm, v0)  # ax + by + cz + d = 0, (a, b, c) derived from norm vector
    t = - (ti.dot(norm, ray_origin) + d) / (ti.dot(norm, ray_direction))

    # Special Case2: triangle behind the ray direction
    if t < 0:
        return False, None

    # Special Case3: entry point out of the triangle area
    entry_point = ray_origin + ray_direction * t

    def inner_angle_check(o, other1, other2, p):
        edge1_norm = (other1 - o).norm()
        edge2_norm = (other2 - o).norm()
        op_norm = (p - o).norm()
        if ti.dot(op_norm, edge1_norm) < ti.dot(edge2_norm, edge1_norm):
            return False
        else:
            return True

    if not inner_angle_check(v0, v1, v2, entry_point):
        return False, None
    if not inner_angle_check(v1, v0, v2, entry_point):
        return False, None
    if not inner_angle_check(v2, v0, v1, entry_point):
        return False, None

    info = {
        't': t,
    }
    return True, info

@ti.func
class CollisionConstraints:

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