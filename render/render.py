import open3d as o3d
import json
import time, os

pause = False

class Render:
    def __init__(self, objs, saving: bool, saving_folder, subdiv=True, subdiv_time=2, ctr_rotate=0.0):
        """
        :param objs: dict()
            {
             'obj1_name': (vertices: ti.Vector.field, indices: ti.Vector.field, color: optional)
            }
        :return:
        """

        # init and setup gui
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        # pause
        # self.pause = False

        # set callbacks
        #self.vis.register_key_callback(ord("R"), self.reset_sim)
        #self.vis.register_key_callback(ord(" "), self.pause_sim)  # space

        # mesh dict
        self.meshes = dict()
        self.colors = dict()

        for obj in objs:
            vertices, indices, color = objs[obj]
            # MSS cloth mesh
            V = o3d.utility.Vector3dVector(vertices.to_numpy())
            F = o3d.utility.Vector3iVector(indices.to_numpy())
            mesh = o3d.geometry.TriangleMesh(V, F)
            mesh.paint_uniform_color(color)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            self.meshes[obj] = mesh
            self.colors[obj] = color
            self.vis.add_geometry(mesh)

        # sphere mesh
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=ball_radius, resolution=120)
        # sphere.translate(ball_centers[0])
        # sphere.paint_uniform_color([1.0, 0.4, 0.2])
        # sphere.compute_vertex_normals()
        # vis.add_geometry(sphere)

        # plane mesh
        points = (
                [[i / 10, 0, -1] for i in range(-10, 11)]
                + [[i / 10, 0, 1] for i in range(-10, 11)]
                + [[-1, 0, i / 10] for i in range(-10, 11)]
                + [[1, 0, i / 10] for i in range(-10, 11)]
        )
        lines = [[i, i + 21] for i in range(21)] + [[i + 42, i + 63] for i in range(21)]
        colors = [[0.7, 0.7, 0.7] for i in range(len(lines))]
        ground_plane = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ground_plane.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(ground_plane)

        # render and view options
        self.rdr = self.vis.get_render_option()
        self.rdr.mesh_show_back_face = True
        self.rdr.mesh_show_wireframe = True
        self.rdr.light_on = False
        self.ctr = self.vis.get_view_control()
        self.set_ctr_from_json('./render/view_control.json')
        #self.ctr.set_lookat([0.2, 0.8, -0.2])
        # self.ctr.set_up([0.0, 0.0, 0.0])

        self.saving = saving
        self.update_times = 0
        self.subdiv = subdiv
        self.subdiv_time = subdiv_time
        self.ctr_rotate = ctr_rotate
        if self.saving:
            self.saving_path = "./images/{}/".format(int(time.time())) if saving_folder == None else saving_folder
            print("Saving images to: {}".format(self.saving_path))
            if not os.path.exists(self.saving_path):
                os.makedirs(self.saving_path)

    def set_ctr_from_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        settings = data['trajectory'][0]
        self.ctr.set_lookat(settings['lookat'])
        self.ctr.set_up(settings['up'])
        self.ctr.set_front(settings['front'])
        self.ctr.set_zoom(settings['zoom'])
        self.ctr.change_field_of_view((settings['field_of_view'] - 60 ) / 5)

    def reset_sim(self):
        # init()
        # reset()
        raise NotImplementedError()

    @staticmethod
    def pause_sim(self):
        global pause
        pause = not pause

    @staticmethod
    def get_pause():
        global pause
        return pause

    def update(self, objs):
        """
        :param objs: dict()
            {
             'obj1_name': (vertices: ti.Vector.field, indices)
            }
        :return:
        """
        self.ctr.rotate(self.ctr_rotate, 0)

        for obj in objs:
            V = o3d.utility.Vector3dVector(objs[obj][0].to_numpy())
            F = o3d.utility.Vector3iVector(objs[obj][1].to_numpy())
            mesh = o3d.geometry.TriangleMesh(V, F)
            if obj.startswith('simulated') and self.subdiv:
                mesh = mesh.subdivide_loop(number_of_iterations=self.subdiv_time)
            self.meshes[obj].vertices = mesh.vertices
            self.meshes[obj].triangles = mesh.triangles
            self.meshes[obj].compute_vertex_normals()
            self.meshes[obj].compute_triangle_normals()
            self.meshes[obj].paint_uniform_color(self.colors[obj])
            #self.meshes[obj].filter_smooth_laplacian(number_of_iterations=10)
            
            self.vis.update_geometry(self.meshes[obj])

        self.vis.update_renderer()
        if self.saving:
            self.vis.capture_screen_image("{}{:07d}.png".format(self.saving_path, self.update_times), False)
        self.update_times += 1

