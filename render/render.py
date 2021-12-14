import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time, os
pause = False

class Render:
    def __init__(self, objs):
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
        self.vis.register_key_callback(ord("R"), self.reset_sim)
        self.vis.register_key_callback(ord(" "), self.pause_sim)  # space

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
            self.vis.add_geometry(mesh)
            self.colors[obj] = color


        WINDOW_WIDTH = 1920  # change this if needed
        WINDOW_HEIGHT = 1080  # change this if needed
        img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color=(255, 255, 255))
        # fnt = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', 64)
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("./font.ttf", size=30)
        d.text((800, 950), "Epoch1:", fill=(0, 0, 0), font=font)  # puts text in upper right
        img.save('./temp.png')
        im = o3d.io.read_image("./temp.png")
        self.im = im
        self.vis.add_geometry(im, reset_bounding_box=False)

        # render and view options
        self.rdr = self.vis.get_render_option()
        self.rdr.mesh_show_back_face = True
        # rdr.mesh_show_wireframe = True
        self.ctr = self.vis.get_view_control()
        self.ctr.set_lookat([0.0, 0.55, 0.6])
        self.ctr.set_up([0.0, 1.0, 0.0])
        self.update_times = 0
        self.saving_path = "./images/{}/".format(int(time.time()))
        print("Saving images to: {}".format(self.saving_path))
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)


    def reset_render(self, objs, text, saving_path, update_times=-1):
        if update_times != -1:
            self.update_times = update_times
        self.saving_path = "./images/{}/".format(saving_path)
        print("Saving images to: {}".format(self.saving_path))
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        self.vis.remove_geometry(self.im, reset_bounding_box=False)
        WINDOW_WIDTH = 1920  # change this if needed
        WINDOW_HEIGHT = 1080  # change this if needed
        img = Image.new('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), color=(255, 255, 255))
        # fnt = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', 64)
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("./font.ttf", size=30)
        d.text((800, 50), text, fill=(0, 0, 0), font=font)  # puts text in upper right
        img.save('./temp.png')
        im = o3d.io.read_image("./temp.png")
        self.im = im
        # time.sleep(0.5)
        self.vis.add_geometry(self.im, reset_bounding_box=False)


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
        for obj in objs:
            vert = np.zeros((objs[obj][0].shape[0],3))
            trian = np.zeros((objs[obj][1].shape[0],3))
            for i in range(objs[obj][0].shape[0]):
                vert[i,:] = objs[obj][0][i]
            for i in range(objs[obj][1].shape[0]):
                trian[i, :] = objs[obj][1][i]

            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert), o3d.utility.Vector3iVector(trian))
            mesh = mesh.subdivide_loop(number_of_iterations=2)
            self.meshes[obj].vertices = mesh.vertices
            self.meshes[obj].triangles = mesh.triangles
            self.meshes[obj].compute_vertex_normals()
            self.meshes[obj].compute_triangle_normals()
            self.meshes[obj].paint_uniform_color(self.colors[obj])
            # self.meshes[obj].filter_smooth_laplacian(number_of_iterations=10)
            self.vis.update_geometry(self.meshes[obj])



        self.vis.update_renderer()
        self.vis.capture_screen_image("{}{:07d}.png".format(self.saving_path, self.update_times), False)
        self.update_times += 1


