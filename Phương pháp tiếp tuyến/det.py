import numpy as np


A = np.array([
    [-2, 3, 4, 3],
    [9, -10, 5, -7],
    [6, 7, 5, 4],
    [10, 9, -2, -10]
])

# Chọn vị trí phần tử (i, j) để tính định thức con (minor)
i, j = 0, 0  # hàng 0, cột 0 (tương ứng M_11)

# Loại bỏ hàng i và cột j
A_minor = np.delete(np.delete(A, i, axis=0), j, axis=1)

# Tính định thức của ma trận con
minor = np.linalg.det(A_minor)

print("Ma trận con A_ij:")
print(A_minor)
print(f"Định thức con M_{i+1}{j+1} =", minor)
