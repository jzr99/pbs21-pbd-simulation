from lib.mesh import *

class Module:
    def __init__(self) -> None:
        self.static_objects = []
        self.simulated_objects = []
        self.inverse_mass = []
        self.constraints = []
        self.collision_constraints = []

    def add_static_objects(self, object: Mesh):
        self.static_objects.append(object)
    
    def add_simulated_objects(self, object: Mesh):
        self.simulated_objects.append(object)

    def add_inverse_mass(self, inv_mass: float):
        self.inverse_mass.append(inv_mass)

    def add_constraints(self, constraint):
        self.constraints.append(constraint)

    def add_collision_constraints(self, collision_constraint):
        self.collision_constraints.append(collision_constraint)

    @ti.func
    def reset_estimate_potions(self):
        for mesh in self.simulated_objects:
            mesh.reset_estimated_vertices()
    

