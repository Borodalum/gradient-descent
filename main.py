from gradient_descent import gradient_descent
import numpy as np
from scipy.optimize import minimize
from plot_graphs import plot_graphs

# Определение функции x^2 * y^2 * log(x^2 + y^2)
def f1(x, y):
    return x ** 2 * y ** 2 * np.log(x ** 2 + y ** 2)

# Определение градиента функции x^2 * y^2 * log(x^2 + y^2)
def gradient1(x, y):
    df_dx = 2 * x * y ** 2 * (np.log(x ** 2 + y ** 2) + x ** 2 / (x ** 2 + y ** 2))
    df_dy = 2 * x ** 2 * y * (np.log(x ** 2 + y ** 2) + y ** 2 / (x ** 2 + y ** 2))
    return np.array([df_dx, df_dy])

# Определение функции x^2 * y^2 * log(8x^2 + 3y^2)
def f2(x, y):
    return x ** 2 * y ** 2 * np.log(8*x**2 + 3*y**2)

# Определение градиента функции x^2 * y^2 * log(8x^2 + 3y^2)
def gradient2(x, y):
    df_dx = 2*x*y**2 * (np.log(8*x**2 + 3*y**2) + 8*x**2/(8*x**2 + 3*y**2))
    df_dy = 2*x**2 * y * (np.log(8*x**2 + 3*y**2) + 3*y**2/(8*x**2 + 3*y**2))
    return np.array([df_dx, df_dy])

# Начальная точка
x0 = 0.1
y0 = 0.1

# Коэффициент
def learning_rate(f, x, y, grad, epsilon):
    return 0.1

# Вызов функции градиентного спуска для функции x^2 * y^2 * log(x^2 + y^2)
x_opt1, y_opt1, num_iterations1, execution_time1 = gradient_descent(f1, gradient1, x0, y0)

print("Функция x^2 * y^2 * log(x^2 + y^2):")
print(f"Критерий останова: |delta f| < 1e-8")
print(f"Число итераций: {num_iterations1}")
print(f"Полученная точка: ({x_opt1[-1]}, {y_opt1[-1]})")
print(f"Полученное значение функции: {f1(x_opt1[-1], y_opt1[-1])}")
print(f"Время работы: {execution_time1:.4f} сек")
plot_graphs(f1, x_opt1, y_opt1)

# Вызов функции градиентного спуска для функции x^2 * y^2 * log(8x^2 + 3y^2)
x_opt2, y_opt2, num_iterations2, execution_time2 = gradient_descent(f2, gradient2, x0, y0)

print("\nФункция x^2 * y^2 * log(8x^2 + 3y^2):")
print(f"Критерий останова: |delta f| < 1e-8")
print(f"Число итераций: {num_iterations2}")
print(f"Полученная точка: ({x_opt2[-1]}, {y_opt2[-1]})")
print(f"Полученное значение функции: {f2(x_opt2[-1], y_opt2[-1])}")
print(f"Время работы: {execution_time2:.4f} сек")
plot_graphs(f2, x_opt2, y_opt2)

def f1_sp(x):
    return f1(x[0], x[1])

def f2_sp(x):
    return f2(x[0], x[1])

# Сравнение результатов с функцией minimize из библиотеки scipy
res1 = minimize(f1_sp, [x0, y0], method='Nelder-Mead', tol=1e-8)
res2 = minimize(f2_sp, [x0, y0], method='Nelder-Mead', tol=1e-8)

print("\nФункция x^2 * y^2 * log(x^2 + y^2):")
print(f"Полученная точка: ({res1.x[0]}, {res1.x[1]})")
print(f"Полученное значение функции: {f1(res1.x[0], res1.x[1])}")

print("\nФункция x^2 * y^2 * log(8x^2 + 3y^2):")
print(f"Полученная точка: ({res2.x[0]}, {res2.x[1]})")
print(f"Полученное значение функции: {f2(res2.x[0], res2.x[1])}")