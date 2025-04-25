import sympy as sp

def tieptuyenmethod(f, g, x0, y0, n):
    # Khai báo biến ký hiệu
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # Tính đạo hàm riêng phần
    dfx = sp.diff(f, x)
    dfy = sp.diff(f, y)
    dgx = sp.diff(g, x)
    dgy = sp.diff(g, y)

    # Chuyển thành hàm có thể tính toán số học
    f_func = sp.lambdify((x, y), f, 'numpy')
    dfx_func = sp.lambdify((x, y), dfx, 'numpy')
    dfy_func = sp.lambdify((x, y), dfy, 'numpy')
    g_func = sp.lambdify((x, y), g, 'numpy')
    dgx_func = sp.lambdify((x, y), dgx, 'numpy')
    dgy_func = sp.lambdify((x, y), dgy, 'numpy')

    for _ in range(n):  # Chạy đúng n lần
        # Tính giá trị hiện tại
        f_val = f_func(x0, y0)
        g_val = g_func(x0, y0)
        dfx_val = dfx_func(x0, y0)
        dfy_val = dfy_func(x0, y0)
        dgx_val = dgx_func(x0, y0)
        dgy_val = dgy_func(x0, y0)

        # Tính mẫu số của công thức
        denominator = dfx_val * dgy_val - dgx_val * dfy_val
        if denominator == 0:
            raise ValueError("Mẫu số bằng 0, phương pháp tiếp tuyến không hội tụ.")

        # Cập nhật giá trị x0, y0
        x0_new = x0 + (-f_val * dgy_val + g_val * dfy_val) / denominator
        y0_new = y0 + (-dfx_val * g_val + dgx_val * f_val) / denominator

        x0, y0 = x0_new, y0_new

    return x0, y0

# Ví dụ sử dụng
x, y = sp.symbols('x y')
f = x + 3 * sp.log(x,10) - y**2
g = 2 * (x**2) - x * y - 5 * x + 1

x_eq, y_eq = tieptuyenmethod(f, g, x0=3, y0=2, n=50)
print(x_eq, y_eq)
