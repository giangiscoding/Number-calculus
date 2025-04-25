#Nguyễn Trường Giang - 2022704
#Phương Pháp chia đôi
#Giải tích số ngày 20/03/2025

import numpy as np
from scipy.optimize import bisect, fsolve

def bisection_method(f, a, b, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("Điều kiện f(a) * f(b) < 0 không thỏa mãn. Hãy chọn lại khoảng chứa nghiệm.")
    
    iter_count = 0
    while iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c  # c là nghiệm chính xác
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    tol= (b - a) / 2
    return (a + b) / 2, tol  # Trả về nghiệm xấp xỉ


# Ví dụ sử dụng:
def func(x):
    return np.sin(x) - 0.5  # Hàm cần tìm nghiệm

print(bisection_method(func, 1,3,100))
