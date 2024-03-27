import numpy as np
import time

def gradient_descent(f, gradient, x0, y0, learnig_rate, epsilon=1e-6, num_iterations=1000):
    x_k = x0
    y_k = y0

    start_time = time.time()
    for i in range(num_iterations):
        grad = gradient(x_k, y_k)
        x_k1 = x_k - learnig_rate * grad[0]
        y_k1 = y_k - learnig_rate * grad[1]
        f_k1 = f(x_k1, y_k1)
        delta_f = abs(f_k1 - f(x_k, y_k))

        if delta_f < epsilon:
            break
        x_k, y_k = x_k1, y_k1

    end_time = time.time()
    execution_time = end_time - start_time

    return x_k1, y_k1, f_k1, i, execution_time
