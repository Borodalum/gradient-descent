import pandas as pd
from coordinate_descent import coordinate_descent
from gradient_descent import gradient_descent
import numpy as np
from scipy.optimize import minimize

from plot_graphs import plot_graphs
from gradient_descent_multiple_variable import gradient_descent_mult


# Определение функции x^2 + y^2
def f1(x, y):
    return x ** 2 + y ** 2


# Определение градиента функции x^2 + y^2
def gradient1(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])


# Определение функции x^2 + 1000 * y^2
def f2(x, y):
    return x ** 2 + 100 * y ** 2


# Определение градиента функции x^2 + 1000 * y^2
def gradient2(x, y):
    df_dx = 2 * x
    df_dy = 200 * y
    return np.array([df_dx, df_dy])


# Начальная точка
x0 = 0.1
y0 = 0.1

learning_rate_const = 0.1


# Коэффициент
def learning_rate(f, x, y, grad, epsilon):
    return learning_rate_const


x_opt1, y_opt1, num_iterations1, execution_time1 = gradient_descent(f1, gradient1, x0, y0, learning_rate)
x_opt2, y_opt2, num_iterations2, execution_time2 = gradient_descent(f2, gradient2, x0, y0, learning_rate)

plot_graphs(f1, x_opt1, y_opt1)
plot_graphs(f2, x_opt2, y_opt2)

df = pd.DataFrame(
    columns=['Method', 'Function', 'Learning Rate', 'Initial Point', 'Optimal Point', 'Function Value', 'Iterations',
             'Execution Time'])

df.loc[0] = ['Gradient Descent', 'x^2 + y^2', f'{learning_rate_const}', f'({x0}, {y0})',
             f'({x_opt1[-1]}, {y_opt1[-1]})', f1(x_opt1[-1], y_opt1[-1]), num_iterations1, execution_time1]
df.loc[1] = ['Gradient Descent', 'x^2 + 100 * y^2', f'{learning_rate_const}', f'({x0}, {y0})',
             f'({x_opt2[-1]}, {y_opt2[-1]})', f2(x_opt2[-1], y_opt2[-1]), num_iterations2, execution_time2]

x_opt1, y_opt1, num_iterations1, execution_time1 = gradient_descent(f1, gradient1, x0, y0)
x_opt2, y_opt2, num_iterations2, execution_time2 = gradient_descent(f2, gradient2, x0, y0)

df.loc[2] = ['Gradient Descent', 'x^2 + y^2', 'Golden section search', f'({x0}, {y0})', f'({x_opt1[-1]}, {y_opt1[-1]})',
             f1(x_opt1[-1], y_opt1[-1]), num_iterations1, execution_time1]
df.loc[3] = ['Gradient Descent', 'x^2 + 100 * y^2', 'Golden section search', f'({x0}, {y0})',
             f'({x_opt2[-1]}, {y_opt2[-1]})', f2(x_opt2[-1], y_opt2[-1]), num_iterations2, execution_time2]


def f1_sp(x):
    return f1(x[0], x[1])


def f2_sp(x):
    return f2(x[0], x[1])


# Сравнение результатов с функцией minimize из библиотеки scipy
res1 = minimize(f1_sp, [x0, y0], method='Nelder-Mead', tol=1e-8)
res2 = minimize(f2_sp, [x0, y0], method='Nelder-Mead', tol=1e-8)

df.loc[4] = ['Scipy Minimize', 'x^2 + y^2', 'Nelder-Mead', f'({x0}, {y0})', f'({res1.x[0]}, {res1.x[1]})',
             f1(res1.x[0], res1.x[1]), res1.nit, res1.nfev]
df.loc[5] = ['Scipy Minimize', 'x^2 + 1000 * y^2', 'Nelder-Mead', f'({x0}, {y0})', f'({res2.x[0]}, {res2.x[1]})',
             f2(res2.x[0], res2.x[1]), res2.nit, res2.nfev]

# Дополнительное задание 1
x_opt1, y_opt1, num_iterations1, execution_time1 = coordinate_descent(f1, gradient1, x0, y0)
x_opt2, y_opt2, num_iterations2, execution_time2 = coordinate_descent(f2, gradient2, x0, y0)

df.loc[6] = ['Coordinate Descent', 'x^2 + y^2', 'Golden section search', f'({x0}, {y0})', f'({x_opt1}, {y_opt1})',
             f1(x_opt1, y_opt1), num_iterations1, execution_time1]
df.loc[7] = ['Coordinate Descent', 'x^2 + 1000 * y^2', 'Golden section search', f'({x0}, {y0})',
             f'({x_opt2}, {y_opt2})', f2(x_opt2, y_opt2), num_iterations2, execution_time2]


# Дополнительное задание 2

# Definition of the function x^3 * y^3 * z^3
def f3(x, y, z):
    return x ** 3 * y ** 3 * z ** 3


# Definition of the gradient of the function x^3 * y^3 * z^3
def gradient3(x, y, z):
    df_dx = 3 * x ** 2 * y ** 3 * z ** 3
    df_dy = 3 * x ** 3 * y ** 2 * z ** 3
    df_dz = 3 * x ** 3 * y ** 3 * z ** 2
    return np.array([df_dx, df_dy, df_dz])


def learning_rate_mult(f, x, grad, epsilon):
    return 0.1


init_points = [1, 1, 1]
points, num_iterations, execution_time = gradient_descent_mult(f3, gradient3, init_points, learning_rate_mult)

df.loc[8] = ['Gradient Descent', 'x^3 * y^3 * z^3', '0.1', f'({init_points[0]}, {init_points[1]}, {init_points[2]})',
             f'({points[-1][0]}, {points[-1][1]}, {points[-1][2]})', f3(points[-1][0], points[-1][1], points[-1][2]),
             num_iterations, execution_time]


# Дополнительное задание 2.2
def f_bad(x, y):
    return x ** 3 + y ** 3 + np.sqrt(x ** 4 + y ** 4)


def gradient_f_bad(x, y):
    df_dx = 3 * x ** 2 + y ** 3 + 4 * x ** 3 / (3 * np.sqrt(x ** 4 + y ** 4))
    df_dy = x ** 3 + 3 * y ** 2 + 4 * y ** 3 / (3 * np.sqrt(x ** 4 + y ** 4))
    return np.array([df_dx, df_dy])


x_bad, y_bad, num_iterations_bad, execution_time_bad = gradient_descent(f_bad, gradient_f_bad, x0, y0)

df.loc[9] = ['Gradient Descent', 'x**3 + y**3 + np.sqrt(x**4 + y**4)', 'Golden section search', f'({x0}, {y0})',
             f'({x_bad[-1]}, {y_bad[-1]})', f_bad(x_bad[-1], y_bad[-1]), num_iterations_bad, execution_time_bad]


def f1_with_noise(x, y):
    return f1(x, y) + np.random.normal(0, 0.000001)


x, y, num_iterations, execution_time = gradient_descent(f1_with_noise, gradient1, x0, y0)

df.loc[10] = ['Gradient Descent', 'x^2 * y^2 * log(x^2 + y^2) with noise', 'Golden section search', f'({x0}, {y0})',
              f'({x[-1]}, {y[-1]})', f1_with_noise(x[-1], y[-1]), num_iterations, execution_time]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df)
