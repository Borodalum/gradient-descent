import numpy as np


def golden_section_search(f, x, y, grad, epsilon):
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    resphi = 2 - phi

    a = -1
    b = 1

    while abs(b - a) > epsilon:
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)

        f1 = f(x - x1 * grad[0], y - x1 * grad[1])
        f2 = f(x - x2 * grad[0], y - x2 * grad[1])

        if f1 < f2:
            b = x2
        else:
            a = x1

    return (a + b) / 2
