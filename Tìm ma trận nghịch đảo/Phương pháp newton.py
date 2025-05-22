import numpy as np

def newton_inv(A, N):
    n = A.shape[0]
    T = np.zeros_like(A, dtype=float)
    C = 1 / A.diagonal()
    np.fill_diagonal(T, C)
    X_0 = T
    E = np.identity(n)
    for i in range(N):
        X_1 = X_0 + X_0 @ (E - A @ X_0)
        X_0 = X_1
    return X_1

A = np.array([
    [10, -1, 3, 1],
    [1, 11, 1, -2],
    [-1, 3, 15, -8],
    [3, 1, 6, 9]
], dtype=float)

A_inv = newton_inv(A, 20)
A_temp = np.linalg.inv(A)

print("Approximate inverse:\n", A_inv)
print("Numpy inverse:\n", A_temp)
print("Error:\n", np.abs(A_inv - A_temp))
