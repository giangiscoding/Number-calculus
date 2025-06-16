import numpy as np

# Ma trận A
A = np.array([
    [50, 3, 4, 3],
    [9, 30, 5, -7],
    [6, 7, -60, 4],
    [10, 9, -2, 100]
], dtype=float)

# Ma trận T = diag(1 / diag(A))
T = np.diag(1 / np.diag(A))

# In ra T với định dạng hợp lý
np.set_printoptions(precision=6, suppress=True)
print("T:")
print(T)

# alpha và beta
alpha = np.eye(A.shape[0]) - T @ A
beta = T

print("\nAlpha:")
print(alpha)

# Khởi tạo x0 là vector ngẫu nhiên trong khoảng [-10, 10], kích thước như A
x0 = np.random.randint(-10, 11, size=(A.shape[0], 1))

# Lặp n lần
n = 50
for i in range(n):
    x1 = alpha @ x0 + beta @ np.ones_like(x0)
    x0 = x1

# Kết quả
print("\nKết quả x sau lặp:")
print(x1)

# So sánh với nghịch đảo A
print("\nMa trận nghịch đảo của A:")
print(np.linalg.inv(A))
