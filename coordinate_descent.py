import time

from golden_section_search import golden_section_search

def coordinate_descent(f, gradient, x0, y0, learning_rate=golden_section_search, epsilon=1e-8, num_iterations=1000):
    x_prev = x0
    y_prev = y0

    start_time = time.time()
    i = 0
    x, y = x_prev, y_prev
    for _ in range(num_iterations):
        grad = gradient(x_prev, y_prev)

        # Оптимизация learning rate для каждого направления
        learning_rate_val = learning_rate(f, x_prev, y_prev, grad, epsilon)
        i += 1

        # Шаг покоординатного спуска с использованием оптимальных learning rate
        x = x_prev - learning_rate_val * grad[0]
        if abs(f(x, y_prev) - f(x_prev, y_prev)) < epsilon:
            break
        x_prev = x

        y = y_prev - learning_rate_val * grad[1]
        if abs(f(x_prev, y) - f(x_prev, y_prev)) < epsilon:
            break
        y_prev = y

    end_time = time.time()
    execution_time = end_time - start_time

    return x, y, i, execution_time