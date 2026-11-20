import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ==================================================
# 1. Statistics
# ==================================================
def statistics_demo():
    print("\n===== STATISTICS =====")

    temps = np.array([15, 16, 18, 21, 24, 27, 29])

    print("Mean:", np.mean(temps))
    print("Minimum:", np.min(temps))
    print("Maximum:", np.max(temps))


# ==================================================
# 2. Error Analysis
# ==================================================
def error_analysis_demo():
    print("\n===== ERROR ANALYSIS =====")

    actual = 29.4
    measured = 29

    absolute_error = abs(actual - measured)
    relative_error = absolute_error / actual

    print("Absolute Error:", absolute_error)
    print("Relative Error:", relative_error)


# ==================================================
# 3. Interpolation
# ==================================================
def interpolation_demo():
    print("\n===== LINEAR INTERPOLATION =====")

    hours = np.array([0, 2, 4, 6, 8, 10, 12])
    temps = np.array([15, 16, 18, 21, 24, 27, 29])

    temp_hour5 = np.interp(5, hours, temps)

    print("Temperature at Hour 5:", temp_hour5)


# ==================================================
# 4. Least Squares Regression
# ==================================================
def regression_demo():
    print("\n===== LEAST SQUARES REGRESSION =====")

    hours = np.array([0, 2, 4, 6, 8, 10, 12])
    temps = np.array([15, 16, 18, 21, 24, 27, 29])

    slope, intercept = np.polyfit(hours, temps, 1)

    print(
        f"Regression Equation: y = {intercept:.2f} + {slope:.2f}x"
    )

    y_fit = intercept + slope * hours

    plt.figure(figsize=(8, 5))
    plt.scatter(hours, temps,
                color='blue',
                label='Observed Data')

    plt.plot(hours,
             y_fit,
             color='red',
             linewidth=2,
             label='Regression Line')

    plt.title("Least Squares Regression")
    plt.xlabel("Hour")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==================================================
# 5. Numerical Differentiation
# ==================================================
def differentiation_demo():
    print("\n===== NUMERICAL DIFFERENTIATION =====")

    hours = np.array([0, 2, 4, 6, 8, 10, 12])
    temps = np.array([15, 16, 18, 21, 24, 27, 29])

    rates = np.diff(temps) / np.diff(hours)

    print("Rates of Change:")
    print(rates)

    plt.figure(figsize=(8, 5))

    plt.plot(hours[:-1],
             rates,
             marker='o')

    plt.title("Rate of Temperature Change")
    plt.xlabel("Hour")
    plt.ylabel("Change per Hour")

    plt.grid(True)
    plt.show()


# ==================================================
# 6. Numerical Integration
# ==================================================
def integration_demo():
    print("\n===== NUMERICAL INTEGRATION =====")

    hours = np.array([0, 2, 4, 6, 8, 10, 12])
    temps = np.array([15, 16, 18, 21, 24, 27, 29])

    area = np.trapz(temps, hours)

    print("Heat Exposure =", area)

    plt.figure(figsize=(8, 5))

    plt.plot(hours,
             temps,
             marker='o')

    plt.fill_between(hours,
                     temps,
                     alpha=0.3)

    plt.title("Numerical Integration")
    plt.xlabel("Hour")
    plt.ylabel("Temperature")

    plt.grid(True)
    plt.show()


# ==================================================
# 7. Matrix Operations
# ==================================================
def matrix_demo():
    print("\n===== MATRIX OPERATIONS =====")

    A = np.array([
        [1, 2],
        [3, 4]
    ])

    x = np.array([5, 6])

    result = A @ x

    print("A =")
    print(A)

    print("x =", x)

    print("Ax =", result)


# ==================================================
# 8. Solving Linear Equations
# ==================================================
def linear_system_demo():
    print("\n===== LINEAR SYSTEM =====")

    A = np.array([
        [2, 3],
        [4, 1]
    ])

    b = np.array([8, 6])

    solution = np.linalg.solve(A, b)

    print("Solution:")
    print("x =", solution[0])
    print("y =", solution[1])


# ==================================================
# 9. Eigenvalues
# ==================================================
def eigen_demo():
    print("\n===== EIGENVALUES =====")

    A = np.array([
        [2, 0],
        [0, 2]
    ])

    values, vectors = np.linalg.eig(A)

    print("Eigenvalues:")
    print(values)

    print("\nEigenvectors:")
    print(vectors)


# ==================================================
# 10. Euler Method
# ==================================================
def euler_demo():
    print("\n===== EULER METHOD =====")

    h = 0.1
    y = 1

    t_values = [0]
    y_values = [1]

    for i in range(20):
        y = y + h * y

        t_values.append((i + 1) * h)
        y_values.append(y)

    plt.figure(figsize=(8, 5))

    plt.plot(
        t_values,
        y_values,
        'ro--',
        label='Euler Approximation'
    )

    plt.title("Euler Method")
    plt.xlabel("t")
    plt.ylabel("y")

    plt.legend()
    plt.grid(True)
    plt.show()


# ==================================================
# 11. solve_ivp
# ==================================================
def solve_ivp_demo():
    print("\n===== SOLVE_IVP =====")

    def model(t, y):
        return -0.5 * y

    t_eval = np.linspace(0, 10, 100)

    solution = solve_ivp(
        model,
        (0, 10),
        [100],
        t_eval=t_eval
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        solution.t,
        solution.y[0],
        color='blue'
    )

    plt.title("Exponential Decay with solve_ivp")
    plt.xlabel("Time")
    plt.ylabel("Amount Remaining")

    plt.grid(True)
    plt.show()


# ==================================================
# 12. Optimization
# ==================================================
def optimization_demo():
    print("\n===== OPTIMIZATION =====")

    x = np.linspace(0, 8, 400)

    y = (x - 4) ** 2 + 1

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        label='f(x)=(x-4)^2+1'
    )

    plt.scatter(
        4,
        1,
        color='red',
        s=100,
        label='Minimum'
    )

    plt.title("Function Optimization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Minimum at x = 4")
    print("Minimum value = 1")


# ==================================================
# 13. Monte Carlo Pi
# ==================================================
def monte_carlo_demo():
    print("\n===== MONTE CARLO PI =====")

    N = 5000

    x = np.random.rand(N)
    y = np.random.rand(N)

    inside = x**2 + y**2 <= 1

    pi_estimate = 4 * np.mean(inside)

    print("Estimated Pi =", pi_estimate)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        x[inside],
        y[inside],
        s=5,
        color='blue'
    )

    plt.scatter(
        x[~inside],
        y[~inside],
        s=5,
        color='red'
    )

    theta = np.linspace(0, np.pi/2, 300)

    plt.plot(np.cos(theta),
             np.sin(theta),
             color='black')

    plt.title(
        f"Monte Carlo π ≈ {pi_estimate:.5f}"
    )

    plt.axis('equal')
    plt.grid(True)

    plt.show()


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    statistics_demo()

    error_analysis_demo()

    interpolation_demo()

    matrix_demo()

    linear_system_demo()

    eigen_demo()

    regression_demo()

    differentiation_demo()

    integration_demo()

    euler_demo()

    solve_ivp_demo()

    optimization_demo()

    monte_carlo_demo()