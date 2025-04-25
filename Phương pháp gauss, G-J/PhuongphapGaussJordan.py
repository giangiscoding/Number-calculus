import numpy as np

def gauss_jordan(A, b):
    augmented_A = np.hstack((A, b.reshape(-1, 2)))
    m = A.shape[0]
    n = A.shape[1]
    m_pivot = -1
    n_pivot = -1
    C = [] # danh sách lưa vị trí của các phần tử pivot đã dùng
    for i in range(m):
        m_pivot, n_pivot, pivot = find_pivot(A)
        print(m_pivot, n_pivot)
        if pivot == 0:
            return augmented_A
        
        # biến đổi hàng sao cho thu được 1 cột có các phần tử bằng 0 và giữ nguyên pivot
        for j in range(m):
            if j == m_pivot:
                continue
            factor = augmented_A[j, n_pivot] / pivot
            augmented_A[j] = augmented_A[j] - factor * augmented_A[m_pivot]
        print(augmented_A)

        # đưa pivot sử dụng vào đúng vị trí của nó
        # nó đáng có khi chuyển đổi thành ma trận đường chéo
        augmented_A[[m_pivot,n_pivot]] = augmented_A[[n_pivot, m_pivot]]
        m_pivot = n_pivot #cập nhật lại ví trí cho pivot
        C.append([m_pivot, n_pivot]) #thêm vị trí pivot vào danh sách

        # tạo lại ma trận A và cho những hàng cột của pivot đã sử dụng bằng 0 để tránh sử dụng lại
        A = np.delete(augmented_A, n, axis=1)
        for i in range(len(C)):
            D = C[i]
            A[D[0]] = 0
            A[:,D[1]] = 0
    return augmented_A

def find_pivot(A):
    m = A.shape[0]
    n = A.shape[1]
    m_pivot = 0
    n_pivot = 0
    for i in range(m):
        for j in range(n):
            if A[i, j] in (1,-1):
                m_pivot = i
                n_pivot = j
                return m_pivot, n_pivot, A[m_pivot, n_pivot]
            if np.abs(A[m_pivot, n_pivot]) < np.abs(A[i, j]):
                m_pivot = i
                n_pivot = j
    return m_pivot, n_pivot, A[m_pivot, n_pivot]

A = np.array([
    [3, -2, 5, -7, 4],
    [2, 9, 14, -30, 0],
    [5, -4, 18, -26, 14],
    [-4, 2, 3, -5, 2],
    [1, 3, 2, -6, -2]
    ])
b = np.array([[3, 5],
              [-5, 10],
              [7, 11],
              [-2, -4],
              [-2, 3]])
print(gauss_jordan(A, b))
