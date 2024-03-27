from gradient_descent import gradient_descent
import numpy as np
# Определение функции x^2 * y^2
def f1(x, y):
    return x ** 2 * y ** 2

# Определение градиента функции x^2 * y^2
def gradient1(x, y):
    df_dx = 2 * x * y ** 2
    df_dy = 2 * x ** 2 * y
    return np.array([df_dx, df_dy])

# Определение функции x^2 + y^2
def f2(x, y):
    return x ** 2 + y ** 2

# Определение градиента функции x^2 + y^2
def gradient2(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Начальная точка
x0 = 1
y0 = 1

# Коэффициент
a_k = 0.1

# Вызов функции градиентного спуска для функции x^2 * y^2
x_opt1, y_opt1, f_opt1, num_iterations1, execution_time1 = gradient_descent(f1, gradient1, x0, y0, a_k)

print("Функция x^2 * y^2:")
print(f"Критерий останова: |delta f| < 1e-8")
print(f"Число итераций: {num_iterations1}")
print(f"Полученная точка: ({x_opt1}, {y_opt1})")
print(f"Полученное значение функции: {f_opt1}")
print(f"Время работы: {execution_time1:.4f} сек")

# Вызов функции градиентного спуска для функции x^2 + y^2
x_opt2, y_opt2, f_opt2, num_iterations2, execution_time2 = gradient_descent(f2, gradient2, x0, y0, a_k)

print("\nФункция x^2 + y^2:")
print(f"Критерий останова: |delta f| < 1e-8")
print(f"Число итераций: {num_iterations2}")
print(f"Полученная точка: ({x_opt2}, {y_opt2})")
print(f"Полученное значение функции: {f_opt2}")
print(f"Время работы: {execution_time2:.4f} сек")
