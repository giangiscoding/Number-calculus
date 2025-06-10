import numpy as np

A = np.array([[4, 5, 2],
              [2, 5, 1],
              [2, 1, 6]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Giá trị riêng:", eigenvalues)
print("Vector riêng (cột):\n", eigenvectors)