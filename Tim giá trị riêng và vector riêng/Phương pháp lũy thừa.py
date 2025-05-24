import numpy as np

def luythua(A,K,tol):
    """
    Phương pháp này dùng để tìm giá trị riêng trội
    A là ma trận cần tìm giá trị riêng trội
    K là số lần lặp tối đa
    tol là sai số cho phép
    """
    n = A.shape[1]
    x = np.ones((n,1))
    check = 0
    lambla = []
    v = np.zeros((n,0))
    k = 1
    # trường hợp có 1 giá trị riêng trội
    y1 = x
    while check == 0 & k <= K:
        y1 = A@y1
        y2 = A@y1
        m = m + 1
        check = kiemtrasongsong(y1,y2,tol)

    if check == 1:
        mask = (y2 != 0) & (y1 != 0)
        ratio = np.divide(y2[mask], y1[mask])
        mean_value = np.mean(ratio) if ratio.size > 0 else 0
        lambla.append(mean_value)
        v = y1 / np.linalg.norm(y1, 2)
    return lambla, v

    # trường hợp có giá trị riêng trội đối nhau
    if check == 0:
        m = 1
        y1 = x
        while check == 0 & m<=m:
            y1 = A@y1
            y2 = A@y1
            y3 = A@y2
            m = m + 1
            check = kiemtrasongsong(y1,y2,tol)
    if check == 1:
        mask = (y3 != 0) & (y2 != 0)
        ratio = np.divide(y3[mask], y2[mask])
        mean_value = np.mean(ratio) if ratio.size > 0 else 0
        lambla.append(mean_value)
    return lambla,v



def kiemtrasongsong(y1,y2,tol):
    check = 0
    cross_product = np.cross(y1, y2)
    np.linalg.norm(cross_product) < tol
    return check