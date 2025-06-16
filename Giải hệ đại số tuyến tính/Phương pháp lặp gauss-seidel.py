import numpy as np

def lap_gauss_seidel_python(A_matrix, b_vector, x0_initial, k_iterations):
    """
    Thực hiện phương pháp lặp Gauss-Seidel để giải hệ phương trình Ax = b.

    Tham số:
        A_matrix (np.ndarray): Ma trận hệ số A (vuông).
        b_vector (np.ndarray): Vector vế phải b.
        x0_initial (np.ndarray): Vector nghiệm khởi tạo x0.
        k_iterations (int): Số lần lặp tối đa.

    Trả về:
        X_history (np.ndarray): Ma trận lưu trữ lịch sử các vector nghiệm qua các lần lặp.
                                 Mỗi cột là một vector nghiệm, cột đầu tiên là x0_initial.
    """
    A = np.array(A_matrix, dtype=float)
    b = np.array(b_vector, dtype=float)
    x0 = np.array(x0_initial, dtype=float)

    n_size = A.shape[0]
    if A.shape[1] != n_size:
        raise ValueError("Ma trận A phải là ma trận vuông.")
    if b.shape[0] != n_size or x0.shape[0] != n_size:
        raise ValueError("Kích thước của b hoặc x0 không khớp với A.")

    # Kiểm tra các phần tử trên đường chéo chính
    diagonal_A = np.diag(A)
    if np.any(diagonal_A == 0):
        # Mặc dù code MATLAB không kiểm tra, nhưng T sẽ có inf nếu diag(A) có 0.
        # Gauss-Seidel chuẩn yêu cầu A(i,i) != 0.
        print("Cảnh báo: Có phần tử bằng 0 trên đường chéo chính của A. "
              "Phương pháp Gauss-Seidel có thể không hội tụ hoặc gây lỗi chia cho 0.")
        # Nếu muốn tiếp tục như MATLAB (có thể lỗi):
        # diagonal_A[diagonal_A == 0] = np.finfo(float).eps # thay thế 0 bằng số rất nhỏ
        # Hoặc raise lỗi:
        # raise ValueError("Phần tử trên đường chéo chính bằng 0, không thể tính T.")

    # Tính T, B_mat, d_vec theo logic của MATLAB
    # T = diag(1./diag(A))
    T_mat = np.diag(1.0 / diagonal_A)

    # B = eye(n) - T*A
    # B_mat[i,i] sẽ bằng 0
    # B_mat[i,j] = -A[i,j]/A[i,i] khi i != j
    B_mat = np.eye(n_size) - T_mat @ A

    # d = T*b
    d_vec = T_mat @ b

    # Khởi tạo ma trận lưu lịch sử nghiệm
    # Số cột là k_iterations + 1 (để chứa cả x0)
    X_history = np.zeros((n_size, k_iterations + 1))
    X_history[:, 0] = x0.copy() # Cột đầu tiên là nghiệm khởi tạo

    x_previous_iter = x0.copy()  # x^(k), tương ứng x0 trong vòng lặp MATLAB
    x_current_iter_build = np.zeros(n_size) # x^(k+1) đang được xây dựng, tương ứng x1 trong MATLAB

    for j_iter_count in range(k_iterations):  # Lặp k_iterations lần
        for i_row_index in range(n_size):   # Tính từng thành phần x_current_iter_build[i_row_index]
            # x1(i) = B(i,i:n)*x0(i:n) + B(i,1:i-1)*x1(1:i-1) + d(i);

            # Phần 1: B_mat[i, i:n] @ x_previous_iter[i:n]
            # Chú ý: B_mat[i,i] = 0.
            # np.dot(B_mat[hàng i, từ cột i đến hết], x_previous_iter[từ phần tử i đến hết])
            sum_term1 = np.dot(B_mat[i_row_index, i_row_index:], x_previous_iter[i_row_index:])

            # Phần 2: B_mat[i, 0:i-1] @ x_current_iter_build[0:i-1]
            # (Sử dụng các giá trị đã được cập nhật trong lần lặp j_iter_count này)
            # np.dot(B_mat[hàng i, từ cột 0 đến cột i-1], x_current_iter_build[từ phần tử 0 đến i-1])
            sum_term2 = 0.0
            if i_row_index > 0: # Chỉ tính nếu có các phần tử trước i_row_index
                sum_term2 = np.dot(B_mat[i_row_index, :i_row_index], x_current_iter_build[:i_row_index])
            
            x_current_iter_build[i_row_index] = sum_term1 + sum_term2 + d_vec[i_row_index]
        
        X_history[:, j_iter_count + 1] = x_current_iter_build.copy()
        x_previous_iter = x_current_iter_build.copy() # Chuẩn bị cho lần lặp tiếp theo

    return X_history

if __name__ == '__main__':
    # Ví dụ sử dụng
    A = np.array([[10, -1,  2,  0],
                          [-1, 11, -1,  3],
                          [ 2, -1, 10, -1],
                          [ 0,  3, -1,  8]])

    b = np.array([6, 25, -11, 15])

    x0 = np.array([0, 0, 0, 0]) # Nghiệm khởi tạo

    k_max = 10 # Số lần lặp

    print("Ma trận A:\n", A)
    print("Vector b:\n", b)
    print("Nghiệm khởi tạo x0:\n", x0)
    print(f"Số lần lặp k = {k_max}\n")

    try:
        X_all_iterations = lap_gauss_seidel_python(A, b, x0, k_max)

        print("Lịch sử các nghiệm (mỗi cột là một nghiệm):")
        for i in range(X_all_iterations.shape[1]):
            print(f"x({i}): {X_all_iterations[:, i]}")

        print("\nNghiệm cuối cùng (sau {} lần lặp):".format(k_max))
        print(X_all_iterations[:, -1])

        # Kiểm tra lại bằng nghiệm chính xác (nếu biết)
        x_exact = np.linalg.solve(A, b)
        print("\nNghiệm chính xác (từ np.linalg.solve):")
        print(x_exact)
        print("Sai số so với nghiệm chính xác:", np.linalg.norm(X_all_iterations[:, -1] - x_exact))

    except ValueError as e:
        print(f"Lỗi: {e}")