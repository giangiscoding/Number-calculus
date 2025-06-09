import numpy as np

A = np.array([
    [-2,  1,  1,  1],
    [-7, -5, -2, -1],
    [ 0, -1, -3, -2],
    [-1,  0, -1,  0]
])


# So sánh với numpy để kiểm tra
print("\nKết quả từ numpy.linalg.eig:")
eigvals, eigvecs = np.linalg.eig(A)
print("Giá trị riêng:", eigvals)
print("Vector riêng:", eigvecs)