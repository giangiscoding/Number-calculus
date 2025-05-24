import numpy as np

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """
    Giải hệ phương trình phi tuyến bằng phương pháp lặp đơn
    Args:
        g: Hàm vector g(x) thỏa mãn x = g(x)
        x0: Điểm ban đầu (numpy array)
        tol: Sai số cho phép
        max_iter: Số lần lặp tối đa
    Returns:
        x: Nghiệm gần đúng
        iter: Số lần lặp thực hiện
    """
    x_old = x0.copy()
    for i in range(max_iter):
        x_new = g(x_old)  # Cập nhật giá trị mới
        error = np.linalg.norm(x_new - x_old)  # Tính sai số
        print(f"Iter {i+1}: x = {x_new}, error = {error:.6f}")
        
        if error < tol:
            print(f"Hội tụ sau {i+1} lần lặp!")
            return x_new, i+1
        
        x_old = x_new  # Cập nhật cho lần lặp tiếp theo
    
    print(f"Không hội tụ sau {max_iter} lần lặp!")
    return x_old, max_iter

# Ví dụ: Giải hệ phương trình x = cos(y), y = sin(x)
def g(x):
    return np.array([np.cos(x[1]), np.sin(x[0])])

# Điểm ban đầu và chạy thuật toán
x0 = np.array([0.5, 0.5])  # Khởi tạo giá trị ban đầu
solution, num_iter = fixed_point_iteration(g, x0, tol=1e-6)

print("\nNghiệm gần đúng:", solution)