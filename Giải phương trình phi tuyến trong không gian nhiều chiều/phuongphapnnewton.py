import numpy as np

def newton_raphson(F, J, x0, tol=1e-6, max_iter=50):
    """
    Giải hệ phương trình phi tuyến bằng phương pháp Newton-Raphson
    Args:
        F: Hàm vector F(x) = 0
        J: Hàm tính ma trận Jacobian của F(x)
        x0: Điểm ban đầu (numpy array)
        tol: Sai số cho phép
        max_iter: Số lần lặp tối đa
    Returns:
        x: Nghiệm gần đúng
        iter: Số lần lặp
    """
    x = x0.copy()
    for i in range(max_iter):
        F_val = F(x)
        J_val = J(x)
        
        # Kiểm tra Jacobian có khả nghịch
        if np.linalg.det(J_val) == 0:
            raise ValueError("Ma trận Jacobian suy biến!")
        
        delta_x = np.linalg.solve(J_val, -F_val)  # Giải hệ J*Δx = -F
        x += delta_x
        
        residual = np.linalg.norm(F_val)
        print(f"Iter {i+1}: x = {x}, Residual = {residual:.6f}")
        
        if residual < tol:
            print(f"Hội tụ sau {i+1} lần lặp!")
            return x, i+1
    
    print(f"Không hội tụ sau {max_iter} lần lặp!")
    return x, max_iter

# ---------------------------------------------------
# Ví dụ: Giải hệ phương trình
# F1: x^2 + y^2 - 4 = 0
# F2: e^x + y - 1 = 0
# ---------------------------------------------------

# Định nghĩa hàm F(x)
def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        np.exp(x[0]) + x[1] - 1
    ])

# Định nghĩa Jacobian của F(x)
def J(x):
    return np.array([
        [2*x[0], 2*x[1]],          # Đạo hàm F1 theo x và y
        [np.exp(x[0]), 1]          # Đạo hàm F2 theo x và y
    ])

# Điểm ban đầu và chạy thuật toán
x0 = np.array([1.0, 1.0])
solution, num_iter = newton_raphson(F, J, x0, tol=1e-6)

print("\nNghiệm gần đúng:", solution)