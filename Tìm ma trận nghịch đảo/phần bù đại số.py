import numpy as np

# Định nghĩa ma trận A
A = np.array([
    [-2, 3, 4, 3],
    [9, -10, 5, -7],
    [6, 7, 5, 4],
    [10, 9, -2, -10]
])

print("Ma trận A ban đầu:")
print(A)

## Phần 1: Tính phần bù đại số A13
# Tạo ma trận con A1 bằng cách xóa hàng 1 và cột 3
A1 = np.delete(A, 0, axis=0)  # Xóa hàng 1
A1 = np.delete(A1, 2, axis=1)  # Xóa cột 3 (index 2 vì đã xóa hàng trước)
print("\nMa trận con A1 (đã xóa hàng 1, cột 3):")
print(A1)

# Tính định thức của A1 (M13) và phần bù đại số A13
M13 = np.linalg.det(A1)
A13 = (-1)**(1+3) * M13
print("\nM13 = det(A1) =", M13)
print("A13 = (-1)^(1+3)*M13 =", A13)

## Phần 2: Tính ma trận nghịch đảo bằng phương pháp phần phụ đại số
print("\n\nBắt đầu tính ma trận nghịch đảo:")
print("Ma trận A:")
print(A)

# Tính định thức của A
detA = np.linalg.det(A)
print("\nĐịnh thức của A (detA) =", detA)

# Tạo ma trận phần phụ đại số Aph
Aph = np.zeros_like(A, dtype=float)  # Khởi tạo ma trận cùng kích thước với A

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        # Tạo ma trận con B bằng cách xóa hàng i và cột j
        B = np.delete(A, i, axis=0)
        B = np.delete(B, j, axis=1)
        
        # Tính phần bù đại số
        Aph[i,j] = (-1)**(i+j) * np.linalg.det(B)

print("\nMa trận phần phụ đại số Aph:")
print(Aph)

# Tính ma trận nghịch đảo: Ainv = (1/detA) * chuyển vị của Aph
Ainv1 = (1/detA) * Aph.T
print("\nMa trận nghịch đảo Ainv1:")
print(Ainv1)

# Kiểm tra kết quả bằng cách nhân A với Ainv (nên ra ma trận đơn vị)
print("\nKiểm tra A * Ainv1 (nên xấp xỉ ma trận đơn vị):")
print(np.round(A @ Ainv1, 10))  # Làm tròn để dễ quan sát

# So sánh với kết quả từ numpy
Ainv_numpy = np.linalg.inv(A)
print("\nMa trận nghịch đảo từ numpy (để kiểm tra):")
print(Ainv_numpy)