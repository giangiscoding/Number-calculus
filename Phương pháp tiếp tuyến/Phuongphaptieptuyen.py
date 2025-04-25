import sympy as sp
from scipy.optimize import minimize_scalar

def newton_method(f_expr, a, b, x0, max_iter):
    # Chuyển biểu thức SymPy thành hàm có thể tính toán bằng NumPy
    f = sp.lambdify(x, f_expr, 'numpy')
    df1 = sp.lambdify(x, sp.diff(f_expr, x), 'numpy')
    df2 = sp.lambdify(x, sp.diff(f_expr, x, 2), 'numpy')  # Đạo hàm cấp 2

    iter_count = 0
    while iter_count < max_iter:
        x_prev = x0
        if df1(x0) == 0:  # Tránh chia cho 0
            print("Lỗi: df1(x0) = 0, phương pháp Newton không khả thi.")
            return None

        x0 = x0 - f(x0) / df1(x0)
        iter_count += 1  # Cập nhật số vòng lặp

    min_df1 = abs(find_min(df1, a, b))
    max_df2 = abs(find_max(df2, a, b))
    tol = (max_df2 / (2 * min_df1)) * ((x0 - x_prev) ** 2)
    
    return x0, tol  # Trả về nghiệm xấp xỉ và sai số

def find_min(f, a, b):
    res = minimize_scalar(f, bounds=(a, b), method='bounded')
    return res.fun if res.success else float('inf')  # Nếu thất bại, trả về vô cùng

def find_max(f, a, b):
    res = minimize_scalar(lambda x: -f(x), bounds=(a, b), method='bounded')
    return -res.fun if res.success else float('-inf')  # Nếu thất bại, trả về vô cùng âm

# Định nghĩa hàm bằng SymPy
x = sp.Symbol('x')
f_expr = x - sp.sin(x) - 0.5  # Hàm f(x) = sin(x) - 0.5

# Gọi hàm Newton
result = newton_method(f_expr, a=1, b=3, x0=2.5, max_iter=4)
print(result)
