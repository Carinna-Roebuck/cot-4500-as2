import numpy as np

# Question 1 - Neville's Interpolation
def neville_interpolation(x, val, w):
    n = len(x)
    neville = np.zeros((n, n))  # Initialize Neville's table

    # Initialize P[i][0] = f(x_i)
    for i in range(n):
        neville[i][0] = val[i]

    # Compute Neville's table
    for j in range(1, n):
        for i in range(n - j):
            neville[i][j] = ((w - x[i + j]) * neville[i][j - 1] - (w - x[i]) * neville[i + 1][j - 1]) / (x[i] - x[i + j])

    # Final interpolated result
    print(f"\nInterpolated value at x = {w}: {neville[0][n - 1]:.16f}")

# Test Case 1 - Neville's Interpolation
x_values = [1.0, 1.5, 2.0, 2.5]
y_values = [2.7183, 4.4817, 7.3891, 12.1825]
x_to_interpolate = 1.8
# Compute interpolated value
neville_interpolation(x_values, y_values, x_to_interpolate)


# Question 2 & 3 - Newton's Interpolation
def divided_difference_table(x, f):
    n = len(x)
    table = np.zeros((n, n))
    
    # Initialize the first column with f(x_i)
    for i in range(n):
        table[i][0] = f[i]
    
    # Compute divided differences
    for j in range(1, n):
        for i in range(j, n):
            table[i][j] = (table[i][j-1] - table[i-1][j-1]) / (x[i] - x[i-j])

    return table
    
def extract_coefficients(table):
    return [table[i][i] for i in range(len(table))]

def newton_interpolation(x, coeffs, x_val):
    n = len(x)
    approx = coeffs[0]
    term = 1

    for i in range(1, n):
        term *= (x_val - x[i - 1])
        approx += coeffs[i] * term

    return approx


# Test Case 2 & 3 - Newton's Interpolation
x_values = [4.2, 4.4, 4.5, 4.7]
fx_values = [10.1575, 11.2085, 12.1000, 14.0347]

# Compute divided difference table
table = divided_difference_table(x_values, fx_values)

# Extract coefficients
coefficients = extract_coefficients(table)

# Print the required coefficients
print("\nCoefficients of Newton’s Polynomial:")
for i in range(1, 4):  # Only coefficients 1, 2, and 3
    print(f"Coefficient {i}: {coefficients[i]:.15f}")

# Compute the 3rd-degree polynomial approximation at x = 4.6
x_target = 4.6
approximation = newton_interpolation(x_values, coefficients, x_target)

# Print the approximation
print(f"\nNewton Forward, 3rd Polynomial Approximation of ({x_target}):")
print(f"{approximation:.15f}")


# Question 4 - Hermite Interpolation
def hermite_approx_matrix(x, f, df):
    n = len(x)
    m = 2 * n       
    cols = 5

    # Initialize arrays:
    z = np.zeros(m)       # Duplicated x-values
    Q = np.zeros((m, m))  # Full Hermite divided difference table

    # Fill in z and the first column of Q
    for i in range(n):
        z[2*i]     = x[i]
        z[2*i + 1] = x[i]
        Q[2*i][0]     = f[i]
        Q[2*i + 1][0] = f[i]
        # For the repeated node, use the given derivative:
        Q[2*i + 1][1] = df[i]
        if i > 0:
            # For the first occurrence, use the formula for repeated nodes:
            Q[2*i][1] = (Q[2*i][0] - Q[2*i - 1][0]) / (z[2*i] - z[2*i - 1])

    # Compute higher-order divided differences
    for j in range(2, m):
        for i in range(j, m):
            denominator = z[i] - z[i - j]
            if abs(denominator) < 1e-15:
                Q[i][j] = 0.0
            else:
                Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / denominator

    # Step 3: Build final matrix H with dimensions (m x cols)
    # Column 0 will be z, and columns 1 to 4 will be Q[i][0] to Q[i][3]
    H = np.zeros((m, cols))
    for i in range(m):
        H[i][0] = z[i]
        for col in range(1, cols):
            H[i][col] = Q[i][col - 1]
    
    return H

def print_hermite_matrix(H):
    print("\nHermite Polynomial Approximation Matrix:\n")
    for row in H:
        # Print each value in scientific notation with 7 decimals.
        row_str = "[ " + "  ".join(f"{val: .7e}" for val in row) + " ]"
        print(row_str)


# Test Case 4 - Hermite Interpolation
x_values = [2.2, 2.5, 2.8]
f_values = [5.5225, 7.3891, 10.2041]
df_values = [1.0452, 1.2840, 1.7437]

# Build and print the Hermite matrix.
H = hermite_approx_matrix(x_values, f_values, df_values)
print_hermite_matrix(H)


# Question 5 - Cubic Spline Interpolation
# Test Case 5 - Cubic Spline Interpolation
x_data = np.array([1, 3, 5, 7])
f_data = np.array([4, 6, 8, 10])

# Set up the system of equations for cubic splines
n = len(x_data) - 1  # Number of intervals
h = np.diff(x_data)  # Differences between consecutive x-values

# Set up matrix A
A = np.zeros((n+1, n+1))
b = np.zeros(n+1)

# Natural spline boundary conditions (second derivative = 0 at the endpoints)
A[0, 0] = 1
A[n, n] = 1

# Filling the matrix A based on the spline conditions
for i in range(1, n):
    A[i, i-1] = h[i-1]
    A[i, i] = 2 * (h[i-1] + h[i])
    A[i, i+1] = h[i]
    
# Set up vector b (right-hand side of the system)
for i in range(1, n):
    b[i] = 3 * ((f_data[i+1] - f_data[i]) / h[i] - (f_data[i] - f_data[i-1]) / h[i-1])

# Solve for the second derivatives (vector x)
x = np.linalg.solve(A, b)

# Output matrix A, vector b, and vector x
print("\nMatrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nVector x :")
print(x)
