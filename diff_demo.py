import taichi as ti

ti.init()

field = ti.Vector.field(3, dtype=ti.float32, shape=(4), needs_grad=True)

# constraint_sum = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
constraint_sum_vec = ti.field(dtype=ti.float32, shape=(2), needs_grad=True)

def init():
    field[0] = [0, 0, 1]
    field[1] = [0, 1, 0]
    field[2] = [1, 0, 0]
    field[3] = [1, 1, 1]

@ti.kernel
def constraint():
    q = field[3]
    p1, p2, p3 = field[0], field[1], field[2]
    # constraint_sum[0] += (q - p1).dot((p2-p1).cross(p3-p1).normalized())
    constraint_sum_vec[0] = field[0].dot(field[1])
    constraint_sum_vec[1] = field[0].dot(field[2])

init()

constraint_sum_vec.grad[0] = 1
constraint_sum_vec.grad[1] = 1
constraint()
constraint.grad()

print(field.grad[0])
print(field.grad[1])
print(field.grad[2])
print(field.grad[3])
print(constraint_sum_vec.grad[0])