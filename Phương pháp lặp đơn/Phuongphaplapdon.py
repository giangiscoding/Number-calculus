import sympy as sp
from scipy.optimize import minimize_scalar

def lapdon(g_expr, x0, eps, q):
    """
    Phương pháp lặp đơn giải phương trình x = g(x)
    
    Tham số:
    g (function): Hàm lặp x = g(x)
    x0 (float): Giá trị ban đầu
    eps (float): Sai số mong muốn
    q (float): Hệ số co |g'(x)| <= q < 1
    
    Trả về:
    float: Nghiệm gần đúng
    """
    g = sp.lambdify(x, g_expr, 'numpy')

    if q >= 1:
        raise ValueError("q phải < 1 để đảm bảo hội tụ")
    
    x1 = g(x0)
    n = 0  # Đếm số lần lặp (không bắt buộc)
    
    # Điều kiện dừng theo tiêu chuẩn sai số hậu nghiệm
    while abs(x1 - x0) >= eps * (1 - q)/q: 
        x0 = x1
        x1 = g(x0)
        n += 1
    print(f"Số lần lặp: {n}")
    return x1

x = sp.Symbol('x')
g_expr = sp.sin(x) + 0.5

nghiem = lapdon(g_expr, x0=2.5, eps=0.000005, q=0.99)

print("Nghiệm gần đúng:", nghiem)