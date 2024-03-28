import time


def gradient_descent_mult(f, gradient, initial_points, learning_rate, epsilon=1e-8,
                          num_iterations=1000):
    prev_points = initial_points

    start_time = time.time()
    i = 0
    points = [prev_points]
    cur_points = prev_points
    for _ in range(num_iterations):
        grad = gradient(*prev_points)

        # Optimize learning rate for each direction
        learning_rate_val = learning_rate(f, prev_points, grad, epsilon)

        i += 1

        # Gradient descent step using optimal learning rate
        cur_points = [0] * len(prev_points)
        for j in range(len(prev_points)):
            cur_points[j] = prev_points[j] - learning_rate_val * grad[j]
        points.append(cur_points)

        if abs(f(*cur_points) - f(*prev_points)) < epsilon:
            break
        prev_points = cur_points

    end_time = time.time()
    execution_time = end_time - start_time

    return points, i, execution_time
