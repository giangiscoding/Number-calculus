import sympy as sp
from scipy.optimize import minimize_scalar

def chord_method(f_expr, a, b, max_iter):
    x = sp.Symbol('x')
    
    # Chuyển biểu thức SymPy thành hàm có thể tính toán bằng NumPy
    f = sp.lambdify(x, f_expr, 'numpy')
    df1 = sp.lambdify(x, sp.diff(f_expr, x), 'numpy')
    df2 = sp.lambdify(x, sp.diff(f_expr, x, 2), 'numpy')  # Đạo hàm cấp 2
    
    # Kiểm tra độ lồi/lõm để chọn điểm bắt đầu
    min_df2 = find_min(df2, a, b)
    if min_df2 > 0:
        if f(a) < 0:
            x0, d = a, b
        else:
            x0, d = b, a
    else:
        if f(a) < 0:
            x0, d = b, a
        else:
            x0, d = a, b

    iter_count = 0
    while iter_count < max_iter:
        x_prev = x0
        x0 = x0 - ((d - x0) * f(x0)) / (f(d) - f(x0))
        iter_count += 1
    
    min_df1 = abs(find_min(df1, a, b))
    max_df1 = abs(find_max(df1, a, b))
    tol = ((max_df1 - min_df1) / min_df1) * abs(x0 - x_prev) if max_df1 != 0 else 0
    
    return x0, tol  # Trả về nghiệm xấp xỉ và sai số

def find_min(f, a, b):
    res = minimize_scalar(f, bounds=(a, b), method='bounded')
    return res.fun if res.success else float('inf')  # Nếu thất bại, trả về vô cùng

def find_max(f, a, b):
    res = minimize_scalar(lambda x: -f(x), bounds=(a, b), method='bounded')
    return -res.fun if res.success else float('-inf')  # Nếu thất bại, trả về vô cùng âm

x = sp.Symbol('x')
f_expr = sp.sin(x) - 0.5

# Gọi hàm phương pháp dây cung
result = chord_method(f_expr, a=1, b=3, max_iter=10)
print(result)
