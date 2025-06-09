import numpy as np

np.random.seed(0)  # Đặt seed để tái tạo được kết quả

# Bước 1: Tạo ma trận giá trị riêng (diagonal) với 2 giá trị đối nhau trội, còn lại nhỏ hơn
eigvals = np.array([5, -5, 0.5, 0.3])  # có thể chỉnh theo ý bạn
D = np.diag(eigvals)

# Bước 2: Tạo ma trận trực chuẩn ngẫu nhiên P
Q, _ = np.linalg.qr(np.random.randn(4, 4))  # dùng QR để tạo ma trận trực chuẩn

# Bước 3: Tạo ma trận A = Q D Q.T
A = Q @ D @ Q.T

# Kiểm tra: in ma trận và giá trị riêng
print("Ma trận A:\n", A)
print("\nGiá trị riêng của A:\n", np.linalg.eigvals(A))
