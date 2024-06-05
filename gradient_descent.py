import math
import time
from golden_section_search import golden_section_search


def gradient_descent(f, gradient, x0, y0, learning_rate=golden_section_search, epsilon=1e-6, num_iterations=2000):
    x_prev = x0
    y_prev = y0

    start_time = time.time()
    i = 0
    x, y = [x_prev], [y_prev]
    x_cur, y_cur = x_prev, y_prev
    for _ in range(num_iterations):
        grad = gradient(x_prev, y_prev)

        # Оптимизация learning rate для каждого направления
        learning_rate_val = learning_rate(f, x_prev, y_prev, grad, epsilon)

        i += 1

        # Шаг градиентного спуска с использованием оптимальных learning rate
        x_cur = x_prev - learning_rate_val * grad[0]
        y_cur = y_prev - learning_rate_val * grad[1]
        x.append(x_cur), y.append(y_cur)
        if abs(f(x_cur, y_cur) - f(x_prev, y_prev)) < epsilon or math.isnan(abs(f(x_cur, y_cur) - f(x_prev, y_prev))):
            break
        x_prev, y_prev = x_cur, y_cur

    end_time = time.time()
    execution_time = end_time - start_time

    return x, y, i, execution_time
