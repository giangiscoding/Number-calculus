import numpy as np
from numpy.random import randint

# Định nghĩa ma trận A
A = np.array([
    [50, 3, 4, 3],
    [9, 30, 5, -7],
    [6, 7, -60, 4],
    [10, 9, -2, 100]
])

N = 10  # Số lần lặp
n = A.shape[0]  # Kích thước ma trận

## Bước 1: Tạo ma trận đường chéo T
T = np.diag(1 / np.diag(A))
print("Ma trận T (đường chéo nghịch đảo):")
print(T)

## Bước 2: Tạo ma trận lặp alpha
alpha = np.eye(n) - T @ A  # @ là phép nhân ma trận trong Python
print("\nMa trận alpha (I - T*A):")
print(alpha)

## Bước 3: Phân tách alpha thành L (tam giác dưới) và U (tam giác trên)
L = np.zeros((n, n))
for i in range(n):
    L[i, :i] = alpha[i, :i]  # Lấy phần dưới đường chéo

U = alpha - L
print("\nMa trận L (tam giác dưới):")
print(L)
print("\nMa trận U (tam giác trên):")
print(U)

## Bước 4: Khởi tạo ma trận ngẫu nhiên x0
x0 = randint(-10, 11, size=(n, n))
x1 = np.zeros((n, n))  # Ma trận kết quả

print("\nMa trận khởi tạo x0 (ngẫu nhiên):")
print(x0)

## Bước 5: Thực hiện vòng lặp
for k in range(N):
    for i in range(n):
        # Công thức lặp: x1 = L*x1 + U*x0 + T
        x1[i, :] = L[i, :] @ x1 + U[i, :] @ x0 + T[i, :]
    x0 = x1.copy()  # Cập nhật giá trị cho vòng lặp tiếp theo

print("\nKết quả sau", N, "lần lặp:")
print(x1)

## Bước 6: Tính nghịch đảo bằng numpy để kiểm tra
A_inv_numpy = np.linalg.inv(A)
print("\nNghịch đảo tính bằng numpy (để kiểm tra):")
print(A_inv_numpy)

# Tính sai số giữa hai phương pháp
error = np.linalg.norm(x1 - A_inv_numpy)
print("\nSai số so với numpy.linalg.inv:", error)