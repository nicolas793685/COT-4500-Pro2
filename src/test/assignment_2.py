import numpy as np
from scipy.interpolate import CubicSpline

# 1. Neville’s Method
def neville_method(x_vals, y_vals, x):
    n = len(x_vals)
    Q = np.zeros((n, n))
    Q[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((x - x_vals[i + j]) * Q[i, j - 1] + (x_vals[i] - x) * Q[i + 1, j - 1]) / (x_vals[i] - x_vals[i + j])
    
    return Q[0, -1]  # Ensure precision matching expected output

# 2. Compute Forward Differences for Newton's Polynomial
def compute_forward_differences(x_vals, y_vals):
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_vals[i + j] - x_vals[i])

    return diff_table[0, :]

# 3. Newton’s Forward Polynomial Approximations
def newton_forward_approximations(x_vals, y_vals):
    forward_diffs = compute_forward_differences(x_vals, y_vals)
    
    degree_1 = forward_diffs[1]
    degree_2 = forward_diffs[2]
    degree_3 = forward_diffs[3]
    
    return round(degree_1, 15), round(degree_2, 15), round(degree_3, 15)

# 4. Newton's Forward Interpolation for f(7.3)
def newton_forward_interpolation(x_vals, y_vals, x):
    h = x_vals[1] - x_vals[0]  # Step size
    forward_diffs = compute_forward_differences(x_vals, y_vals)
    p = (x - x_vals[0]) / h

    result = y_vals[0] + p * forward_diffs[1]
    result += (p * (p - 1) / 2) * forward_diffs[2]
    result += (p * (p - 1) * (p - 2) / 6) * forward_diffs[3]
    
    return round(result, 15)

# 5. Hermite Interpolation Table
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

    result = np.round(Q, 15)
    output = np.column_stack((np.repeat(x_vals, 2), result))
    return output
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

    return np.round(Q, 12)
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

    return Q

# 6. Cubic Spline Interpolation
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
    
    return A, b, x_vals
    cs = CubicSpline(x_vals, y_vals, bc_type='natural')
    coefficients = cs.c  # No rounding for precision
    return coefficients, x_vals, y_vals

# ================== Running the Implementations ==================
if __name__ == "__main__":
    # Question 1: Neville’s Method
    x_vals1 = np.array([3.6, 3.8, 3.9])
    y_vals1 = np.array([1.675, 1.436, 1.318])
    x_target1 = 3.7
    print(neville_method(x_vals1, y_vals1, x_target1))
    print("\n")


    # Question 2: Newton’s Forward Polynomial Approximations
    x_vals2 = np.array([7.2, 7.4, 7.5, 7.6])
    y_vals2 = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    degree_1, degree_2, degree_3 = newton_forward_approximations(x_vals2, y_vals2)
    print(degree_1)
    print(degree_2)
    print(degree_3)
    print("\n")


    # Question 3: Approximate f(7.3) using degree-3 polynomial
    f_approx_7_3 = newton_forward_interpolation(x_vals2, y_vals2, 7.3)
    print(f_approx_7_3)
    print("\n")


    # Question 4: Hermite Interpolation Table
    x_vals4 = np.array([3.6, 3.8, 3.9])
    y_vals4 = np.array([1.675, 1.436, 1.318])
    derivatives4 = np.array([-1.195, -1.188, -1.182])
    print(hermite_interpolation(x_vals4, y_vals4, derivatives4))
    print("\n")


    # Question 5: Cubic Spline Interpolation
    coefficients, matrix_A, vector_b = cubic_spline_interpolation(np.array([2, 5, 8, 10]), np.array([3, 5, 7, 9]))
    print(coefficients)
    print(matrix_A)
    print(vector_b)
    print("\n")

