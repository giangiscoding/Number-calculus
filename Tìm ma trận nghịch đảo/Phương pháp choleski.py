import numpy as np

# Tạo ma trận A
A = np.array([[-2, 3, 4, 3],
              [9, -10, 5, -7],
              [6, 7, 5, 4],
              [10, 9, -2, -10]])

# Tính A*A'
A = A @ A.T
print("Ma trận A (A*A'):")
print(A)

# Phân tích Cholesky
n = A.shape[0]
Q = np.zeros((n, n))

for i in range(n):
    Q[i, i] = np.sqrt(A[i, i] - np.sum(Q[:i, i]**2))
    if i < n-1:
        Q[i, i+1:] = (1/Q[i, i]) * (A[i, i+1:] - Q[:i, i].T @ Q[:i, i+1:])

print("\nMa trận Q (Cholesky factor):")
print(Q)

print("\nKiểm tra Q'*Q (nên bằng A):")
print(Q.T @ Q)

# Tính nghịch đảo của Q
Qinv = np.diag(1/np.diag(Q))
for i in range(n-1, -1, -1):
    if i < n-1:
        Qinv[i, i+1:] = (-1/Q[i, i]) * Q[i, i+1:] @ Qinv[i+1:, i+1:]

print("\nMa trận Q^{-1}:")
print(Qinv)

print("\nKiểm tra Q*Q^{-1} (nên bằng I):")
print(Q @ Qinv)

# Tính nghịch đảo của A
Ainv = Qinv @ Qinv.T
print("\nMa trận A^{-1} tính từ Q^{-1}:")
print(Ainv)

print("\nKiểm tra A*A^{-1} (nên bằng I):")
print(A @ Ainv)

print("\nNghịch đảo tính bằng numpy.linalg.inv để kiểm tra:")
Ainv_numpy = np.linalg.inv(A)
print(Ainv_numpy)

print("\nSai số giữa hai phương pháp:", np.linalg.norm(Ainv - Ainv_numpy))