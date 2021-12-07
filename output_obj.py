import open3d as o3d

cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=1.0, resolution=10, split=50)
o3d.io.write_triangle_mesh('obj/cylinder.obj', mesh=cylinder)

# box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
# o3d.io.write_triangle_mesh('obj/box.obj', mesh=box)