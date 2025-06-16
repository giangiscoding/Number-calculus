import numpy as np

# Định nghĩa ma trận A
A = np.array([
    [50, 3, 4, 3],
    [9, 30, 5, -7],
    [6, 7, -60, 4],
    [10, 9, -2, 100]
])

# Khởi tạo ma trận xấp xỉ ban đầu x0
# Thường chọn x0 = A^T/(||A||_1 * ||A||_∞) hoặc x0 = k*A^T
x0 = A.T / (np.linalg.norm(A, 1) * np.linalg.norm(A, np.inf))

print("Ma trận A:")
print(A)
print("\nXấp xỉ ban đầu x0:")
print(x0)

# Ma trận đơn vị
E = np.eye(A.shape[0])

# Tính chuẩn vô cùng của sai số ban đầu
error = np.linalg.norm(A @ x0 - E, np.inf)
print(f"\nSai số ban đầu (chuẩn vô cùng): {error:.4e}")

# Thực hiện 3 bước lặp Newton
for i in range(3):
    x1 = x0 - x0 @ (A @ x0 - E)
    x0 = x1
    error = np.linalg.norm(A @ x0 - E, np.inf)
    print(f"\nSau bước lặp {i+1}:")
    print("Ma trận xấp xỉ nghịch đảo:")
    print(x0)
    print(f"Sai số (chuẩn vô cùng): {error:.4e}")

# Kiểm tra kết quả
print("\nKiểm tra A*x0 (nên xấp xỉ ma trận đơn vị):")
print(A @ x0)

# So sánh với nghịch đảo tính bằng numpy
A_inv_numpy = np.linalg.inv(A)
print("\nMa trận nghịch đảo tính bằng numpy:")
print(A_inv_numpy)
print("\nSai số so với numpy.linalg.inv:", np.linalg.norm(x0 - A_inv_numpy))