import taichi as ti


class BoundingBox:
    def __init__(self) -> None:
        self.x_min = ti.field(ti.f32, ())
        self.x_max = ti.field(ti.f32, ())
        self.y_min = ti.field(ti.f32, ())
        self.y_max = ti.field(ti.f32, ())
        self.z_min = ti.field(ti.f32, ())
        self.z_max = ti.field(ti.f32, ())

    def intersect(self, ray_origin, ray_direction):
        # check if from the ray_origin, will the line in the ray_direction intersect with the box
        xn = (self.x_min[None] - ray_origin[0]) / ray_direction[0]
        xf = (self.x_max[None] - ray_origin[0]) / ray_direction[0]
        if xn > xf:
            xn, xf = xf, xn
        yn = (self.y_min[None] - ray_origin[1]) / ray_direction[1]
        yf = (self.y_max[None] - ray_origin[1]) / ray_direction[1]
        if yn > yf:
            yn, yf = yf, yn
        zn = (self.z_min[None] - ray_origin[2]) / ray_direction[2]
        zf = (self.z_max[None] - ray_origin[2]) / ray_direction[2]
        if zn > zf:
            zn, zf = zf, zn
        xy_overlap = xn <= yf and yn <= xf
        xz_overlap = xn <= zf and zn <= xf
        yz_overlap = yn <= zf and zn <= yf

        return xy_overlap and xz_overlap and yz_overlap

    def set_infinity(self):
        self.x_min[None] = float('inf')
        self.x_max[None] = -float('inf')
        self.y_min[None] = float('inf')
        self.y_max[None] = -float('inf')
        self.z_min[None] = float('inf')
        self.z_max[None] = -float('inf')

    def update_bounding_box(self, vertices):
        # clear
        self.set_infinity()

        # enumerate each vertices
        for i in ti.ndrange(*vertices.shape):
            self.x_min[None] = min(self.x_min[None], vertices[i].x)
            self.y_min[None] = min(self.y_min[None], vertices[i].y)
            self.z_min[None] = min(self.z_min[None], vertices[i].z)
            self.x_max[None] = max(self.x_max[None], vertices[i].x)
            self.y_max[None] = max(self.y_max[None], vertices[i].y)
            self.z_max[None] = max(self.z_max[None], vertices[i].z)
