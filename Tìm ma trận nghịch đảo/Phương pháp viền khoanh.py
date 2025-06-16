import numpy as np

def vien_quanh(A, verbose=False):
    """
    Tính nghịch đảo của ma trận vuông A bằng phương pháp viền quanh (bordering method).
    
    Parameters:
        A (np.ndarray): Ma trận vuông cần nghịch đảo.
        verbose (bool): Nếu True, in ra từng bước tính toán.
    
    Returns:
        np.ndarray: Ma trận nghịch đảo của A.
    """
    n = A.shape[0]
    if n == 1:
        if verbose:
            print("\n=== Bước 1: Ma trận 1x1 ===")
            print("A =", A)
            print("A^{-1} =", 1/A[0,0])
        return np.array([[1 / A[0, 0]]])
    
    # Bắt đầu từ ma trận 2x2
    if verbose:
        print("\n=== Bước 1: Khởi tạo với ma trận 2x2 ===")
        print("A_2 =")
        print(A[:2, :2])
    
    a_inv = np.linalg.inv(A[:2, :2])
    
    if verbose:
        print("\nNghịch đảo A_2^{-1}:")
        print(a_inv)
    
    # Mở rộng dần lên ma trận nxn
    for k in range(3, n + 1):
        if verbose:
            print(f"\n=== Bước {k-1}: Mở rộng lên ma trận {k}x{k} ===")
            print(f"A_{k} =")
            print(A[:k, :k])
        
        a_inv = vien_quanh_step(A[:k, :k], a_inv, verbose)
    
    return a_inv

def vien_quanh_step(A_k, a_inv_prev, verbose=False):
    """
    Mở rộng nghịch đảo từ ma trận (k-1)x(k-1) lên kxk.
    
    Parameters:
        A_k (np.ndarray): Ma trận kxk cần tính nghịch đảo.
        a_inv_prev (np.ndarray): Nghịch đảo của ma trận (k-1)x(k-1) con.
        verbose (bool): Nếu True, in ra từng bước tính toán.
    
    Returns:
        np.ndarray: Nghịch đảo của A_k.
    """
    k = A_k.shape[0]
    
    # Tách các thành phần U, V, a_kk
    U = A_k[-1, :-1].reshape(1, -1)  # Hàng cuối (không bao gồm phần tử a_kk)
    V = A_k[:-1, -1].reshape(-1, 1)  # Cột cuối (không bao gồm phần tử a_kk)
    a_kk = A_k[-1, -1]               # Phần tử góc phải dưới
    
    if verbose:
        print("\nCác thành phần:")
        print("U =", U)
        print("V =", V)
        print("a_kk =", a_kk)
        print("\nA_{k-1}^{-1} =")
        print(a_inv_prev)
    
    # Tính các thành phần beta
    U_a_inv = U @ a_inv_prev
    a_inv_V = a_inv_prev @ V
    denominator = a_kk - U_a_inv @ V
    
    if verbose:
        print("\nTính toán trung gian:")
        print("U * A_{k-1}^{-1} =", U_a_inv)
        print("A_{k-1}^{-1} * V =", a_inv_V)
        print("Mẫu số (a_kk - U*A_{k-1}^{-1}*V) =", denominator)
    
    # Tránh chia cho 0 (nếu ma trận không khả nghịch)
    if np.abs(denominator) < 1e-10:
        raise ValueError("Ma trận không khả nghịch (determinant gần bằng 0).")
    
    beta_22 = 1 / denominator
    beta_12 = -a_inv_V * beta_22
    beta_21 = -U_a_inv * beta_22
    beta_11 = a_inv_prev + (a_inv_V @ U_a_inv) * beta_22
    
    if verbose:
        print("\nCác thành phần nghịch đảo:")
        print("beta_11 =")
        print(beta_11)
        print("beta_12 =", beta_12.flatten())
        print("beta_21 =", beta_21.flatten())
        print("beta_22 =", beta_22)
    
    # Ghép các thành phần để tạo ma trận nghịch đảo
    A_inv = np.block([
        [beta_11,       beta_12],
        [beta_21.reshape(1, -1), beta_22]
    ])
    
    if verbose:
        print("\nMa trận nghịch đảo A_{k}^{-1}:")
        print(A_inv)
    
    return A_inv

# Ví dụ sử dụng
if __name__ == "__main__":
    # Test với ma trận 3x3
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ])
    
    print("Ma trận A:")
    print(A)
    
    print("\n=== Bắt đầu tính toán ===")
    A_inv_vienquanh = vien_quanh(A, verbose=True)
    
    print("\n=== Kết quả cuối cùng ===")
    print("\nNghịch đảo tính bằng VienQuanh:")
    print(A_inv_vienquanh)
    
    # Kiểm tra bằng numpy
    A_inv_numpy = np.linalg.inv(A)
    print("\nNghịch đảo tính bằng numpy (để kiểm tra):")
    print(A_inv_numpy)
    
    # Kiểm tra sai số
    error = np.linalg.norm(A_inv_vienquanh - A_inv_numpy)
    print(f"\nSai số so với numpy.linalg.inv: {error:.2e}")