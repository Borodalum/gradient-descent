import numpy as np
import time

from golden_section_search import golden_section_search


def gradient_descent(f, gradient, x0, y0, learning_rate=golden_section_search, epsilon=1e-8, num_iterations=1000):
    x_k = x0
    y_k = y0

    start_time = time.time()
    for i in range(num_iterations):
        grad = gradient(x_k, y_k)

        # Оптимизация learning rate для каждого направления
        learning_rate_x = learning_rate(f, gradient, x_k, y_k, grad, epsilon)
        learning_rate_y = learning_rate(f, gradient, x_k, y_k, grad, epsilon)

        # Шаг градиентного спуска с использованием оптимальных learning rate
        x_k1 = x_k - learning_rate_x * grad[0]
        y_k1 = y_k - learning_rate_y * grad[1]

        if abs(f(x_k1, y_k1) - f(x_k, y_k)) < epsilon:
            break
        x_k, y_k = x_k1, y_k1

    end_time = time.time()
    execution_time = end_time - start_time

    return x_k1, y_k1, i + 1, execution_time
