import numpy as np

A = np.array([
    [ 0.308028,    4.29374056, -1.38633287,  0.15108192],
    [ 4.29374056,  0.31992,     1.21184431,  1.69639799],
    [-1.38633287,  1.21184431, -0.25076267, -0.47251727],
    [ 0.15108192,  1.69639799, -0.47251727,  0.42281467]
])


# So sánh với numpy để kiểm tra
print("\nKết quả từ numpy.linalg.eig:")
eigvals, eigvecs = np.linalg.eig(A)
print("Giá trị riêng:", eigvals)
print("Vector riêng:", eigvecs)