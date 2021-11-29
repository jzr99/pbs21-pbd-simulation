from numpy import inf


class BoudingBox:
    def __init__(self) -> None:
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.z_min = 0
        self.z_max = 0
    
    def intersect(self, ray_origin, ray_direction):
        # check if from the ray_origin, will the line in the ray_direction intersect with the box
        xn = (self.x_min - ray_origin[0]) / ray_direction[0]
        xf = (self.x_max - ray_origin[0]) / ray_direction[0]
        if xn > xf:
            xn, xf = xf, xn
        yn = (self.y_min - ray_origin[1]) / ray_direction[1]
        yf = (self.y_max - ray_origin[1]) / ray_direction[1]
        if yn > yf:
            yn, yf = yf, yn
        zn = (self.z_min - ray_origin[2]) / ray_direction[2]
        zf = (self.z_max - ray_origin[2]) / ray_direction[2]
        if zn > zf:
            zn, zf = zf, zn
        xy_overlap = xn <= yf and yn <= xf
        xz_overlap = xn <= zf and zn <= xf
        yz_overlap = yn <= zf and zn <= yf

        return xy_overlap and xz_overlap and yz_overlap
    
    def set_infinity(self):
        self.x_min = inf
        self.x_max = -inf
        self.y_min = inf
        self.y_max = -inf
        self.z_min = inf
        self.z_max = -inf