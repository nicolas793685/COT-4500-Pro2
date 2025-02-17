import numpy as np
from scipy.interpolate import CubicSpline

def neville_method(x_vals, y_vals, x):
    n = len(x_vals)
    Q = np.zeros((n, n))
    Q[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((x - x_vals[i + j]) * Q[i, j - 1] + (x_vals[i] - x) * Q[i + 1, j - 1]) / (x_vals[i] - x_vals[i + j])
    
    return Q[0, -1]

def compute_forward_differences(x_vals, y_vals):
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_vals[i + j] - x_vals[i])
    
    return diff_table[:n, :n]

def newton_forward_approximations(x_vals, y_vals):
    forward_diffs = compute_forward_differences(x_vals, y_vals)
    degree_1 = forward_diffs[0, 1]
    degree_2 = forward_diffs[0, 2] if len(x_vals) > 2 else 0
    degree_3 = forward_diffs[0, 3] if len(x_vals) > 3 else 0
    
    return round(float(degree_1), 12), round(float(degree_2), 12), round(float(degree_3), 12)

def newton_forward_interpolation(x_vals, y_vals, x):
    h = x_vals[1] - x_vals[0]
    forward_diffs = compute_forward_differences(x_vals, y_vals)
    p = (x - x_vals[0]) / h
    
    result = y_vals[0] + p * forward_diffs[0, 1]
    if len(x_vals) > 2:
        result += (p * (p - 1) / 2) * forward_diffs[0, 2]
    if len(x_vals) > 3:
        result += (p * (p - 1) * (p - 2) / 6) * forward_diffs[0, 3]
    
    return round(float(result), 12)

def hermite_interpolation(x_vals, y_vals, derivatives):
    n = len(x_vals)
    m = 2 * n
    z = np.zeros(m)
    Q = np.zeros((m, m))

    for i in range(n):
        z[2 * i] = x_vals[i]
        z[2 * i + 1] = x_vals[i]
        Q[2 * i, 0] = y_vals[i]
        Q[2 * i + 1, 0] = y_vals[i]
        Q[2 * i + 1, 1] = derivatives[i]
        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, m):
        for i in range(m - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])

    return np.column_stack((np.repeat(x_vals, 2), np.round(Q, 15)))

def cubic_spline_interpolation(x_vals, y_vals):
    n = len(x_vals)
    h = np.diff(x_vals)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y_vals[i + 1] - y_vals[i]) / h[i] - (y_vals[i] - y_vals[i - 1]) / h[i - 1])
    
    coeffs = np.linalg.solve(A, b)
    return np.round(A, 8), np.round(b, 8), x_vals

if __name__ == "__main__":
    x_vals1 = np.array([3.6, 3.8, 3.9])
    y_vals1 = np.array([1.675, 1.436, 1.318])
    x_target1 = 3.7
    print(neville_method(x_vals1, y_vals1, x_target1))
    print("\n")

    x_vals2 = np.array([7.2, 7.4, 7.5, 7.6])
    y_vals2 = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    degree_1, degree_2, degree_3 = newton_forward_approximations(x_vals2, y_vals2)
    print(degree_1)
    print(degree_2)
    print(degree_3)
    print("\n")

    f_approx_7_3 = newton_forward_interpolation(x_vals2, y_vals2, 7.3)
    print(f_approx_7_3)
    print("\n")

    x_vals4 = np.array([3.6, 3.8, 3.9])
    y_vals4 = np.array([1.675, 1.436, 1.318])
    derivatives4 = np.array([-1.195, -1.188, -1.182])
    print(hermite_interpolation(x_vals4, y_vals4, derivatives4))
    print("\n")

    coefficients, matrix_A, vector_b = cubic_spline_interpolation(np.array([2, 5, 8, 10]), np.array([3, 5, 7, 9]))
    print(coefficients)
    print(matrix_A)
    print(vector_b)
    print("\n")
