import taichi as ti

@ti.data_oriented
class Simulation(object):

    def __init__(self, mesh_sphere, mesh_cloth, render):
        # self.timestep = timestep
        self.mesh_sphere = mesh_sphere
        self.mesh_cloth = mesh_cloth
        self.solverIterations = 4
        self.timeStep = 0.003
        self.gravity = 0.981
        self.windSpeed = 1.5
        self.velocityDamping = 0.999
        self.stretchFactor = 0.999
        self.bendFactor = 0.3
        self.wireframe = False
        self.render = render

    def update(self):
        self.simulate()
        self.render.update({'sphere': self.mesh_sphere.export_for_render(), 'cloth': self.mesh_cloth.export_for_render()})
        if not self.render.vis.poll_events():
            exit(-1)
        self.render.vis.update_renderer()

    @ti.kernel
    def simulate(self):

        self.mesh_cloth.apply_impulse(2.0 * self.timeStep * ti.Vector([0.0, -self.gravity, 0.0]))
        # TODO apply wind
        for i in ti.grouped(self.mesh_cloth.velocities):
            self.mesh_cloth.estimated_vertices[i] = self.mesh_cloth.vertices[i] + self.timeStep * self.mesh_cloth.velocities[i]

        # TODO setup constrain
        # TODO project constraint
        # update velocities and positions
        for i in ti.grouped(self.mesh_cloth.velocities):
            self.mesh_cloth.velocities[i] = (self.mesh_cloth.estimated_vertices[i] - self.mesh_cloth.vertices[i]) / self.timeStep
            self.mesh_cloth.vertices[i] = self.mesh_cloth.estimated_vertices[i]

        #  TODO Update velocities of colliding vertices
