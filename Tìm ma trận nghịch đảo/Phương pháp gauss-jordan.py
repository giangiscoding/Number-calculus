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

# Tạo ma trận mở rộng [A|I]
m, n = A.shape
A_bs = np.hstack((A, np.eye(m)))
print("\nMa trận mở rộng [A|I]:")
print(A_bs)

# Thiết lập ngưỡng cho giá trị không
tol = 1e-5

# Khởi tạo các biến
i = 0
j = 0
jb = []

while i < m and j < n:
    # Tìm phần tử có giá trị tuyệt đối lớn nhất trong cột j từ hàng i trở đi
    p = np.max(np.abs(A_bs[i:m, j]))
    k = np.argmax(np.abs(A_bs[i:m, j])) + i
    
    if p < tol:
        # Cột không đáng kể, gán bằng 0
        A_bs[i:m, j] = 0
        j += 1
    else:
        # Lưu chỉ số cột
        jb.append(j)
        
        # Đổi chỗ hàng i và hàng k
        A_bs[[i, k], j:] = A_bs[[k, i], j:]
        
        # Chia hàng pivot cho phần tử pivot
        A_bs[i, j:] = A_bs[i, j:] / A_bs[i, j]
        
        # Trừ bội số của hàng pivot từ các hàng khác
        for k in range(m):
            if k != i:
                A_bs[k, j:] = A_bs[k, j:] - A_bs[k, j] * A_bs[i, j:]
        
        i += 1
        j += 1

print("\nMa trận sau khi áp dụng Gauss-Jordan:")
print(A_bs)

# Trích xuất ma trận nghịch đảo (nửa bên phải)
Ainv2 = A_bs[:, n:]
print("\nMa trận nghịch đảo tính được:")
print(Ainv2)

# Kiểm tra bằng numpy
Ainv_numpy = np.linalg.inv(A)
print("\nMa trận nghịch đảo từ numpy (để kiểm tra):")
print(Ainv_numpy)

# Tính sai số
error = np.linalg.norm(Ainv2 - Ainv_numpy)
print("\nSai số giữa hai phương pháp:", error)