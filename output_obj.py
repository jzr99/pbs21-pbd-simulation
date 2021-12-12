import open3d as o3d

# cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=1.0, resolution=10, split=50)
# o3d.io.write_triangle_mesh('obj/cylinder.obj', mesh=cylinder)

# box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
# o3d.io.write_triangle_mesh('obj/box.obj', mesh=box)


filename1 = './obj/cloth.obj'
filename2 = './obj/cloth_vertical.obj'
with open(filename1, 'r') as f1:
    with open(filename2, 'w') as f2:
        for line in f1.readlines():
            items = line.strip().split(" ")
            if items[0] == 'v':  # vertex
                f2.write(items[0] + " " + items [1] + " " + items[3] + " "  + items[2] + "\n")
            elif items[0] == 'f':
                f2.write(line)
