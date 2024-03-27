def plot_graphs(f, x_opt, y_opt, x_range=(-5, 5), y_range=(-5, 5)):
    import matplotlib.pyplot as plt
    import numpy as np

    # Создание 3D графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(*x_range, 100)
    Y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.scatter(x_opt, y_opt, f(x_opt, y_opt), color='r')  # Траектория метода
    plt.show()

    # Создание 2D графика с линиями уровня
    plt.figure()
    contour = plt.contour(X, Y, Z, cmap='viridis')
    plt.plot(x_opt, y_opt, 'r-')  # Траектория метода
    plt.show()
    plt.close()