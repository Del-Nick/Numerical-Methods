from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

T_0 = (5 + 15 + 32) / 15
x_0 = 5

x_max = 10
t_max = 1


def plot_graphs(T: np.array, x: np.array, t: float,
                save: bool = False, folder: str = None):
    """
    Строит обычные графики в каждый момент времени
    :param T: решение уравнения в заданный момент времени
    :param x: массив с координатами
    :param t: массив со временем
    :param save: Если True, сохраняет графики на диск. Если False, только показывает
    :param folder: Если save=True, должна быть указана папка. Задаётся с функциях с решением уравнений
    :return: None
    """
    plt.figure('Plot Graph')

    plt.title(f't = {t:.3f}')

    plt.plot(x, T)
    plt.xlabel('x, м')
    plt.ylabel('T, К')
    plt.grid()

    if save:
        if folder is not None:
            if not os.path.isdir(folder):
                os.mkdir(folder)
            plt.savefig(f'{folder}/t = {t:.3f}.png')
        else:
            print('Укажи папку для сохранения графиков!')
            exit()
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()


def save_gif(folder: str):
    """
    Собирает все сохранённые картинки, делает гифку и удаляет все графики, кроме нужных моментов времени
    :param folder: Если save=True, должна быть указана папка. Задаётся с функциях с решением уравнений
    :return: None
    """
    files = glob(f'{folder}/*.png')

    frames = []

    for file in files:
        if 't = ' in file.split("\\")[-1]:
            frame = Image.open(file)
            frames.append(frame)

    frames[0].save(f'{folder}/Solution.gif',
                   save_all=True,
                   append_images=frames[1:],
                   optimize=True,
                   duration=50,
                   loop=0)

    times_to_save = [0, 0.1, 0.2, 0.3, 0.5, 1]
    times_to_save = [f't = {times_to_save[i]:.3f}' for i in range(len(times_to_save))]

    pbar = tqdm(total=len(files))
    pbar.set_description('Удаляю лишние графики')

    for file in files:
        if file.split("\\")[-1].replace('.png', '') not in times_to_save:
            if 't = ' in file.split("\\")[-1]:
                os.remove(file)

        pbar.update(1)


def surface_graph(T: np.array, x: np.array, t: np.array,
                  save: bool = False, folder: str = None):
    """
    Строит объёмный график поверхности решения
    :param T: решение уравнения
    :param x: массив с координатами
    :param t: массив со временем
    :param save: Если True, сохраняет графики на диск. Если False, только показывает
    :param folder: Если save=True, должна быть указана папка. Задаётся с функциях с решением уравнений
    :return: None
    """
    plt.figure('Surface Graph')

    X, Y = np.meshgrid(x, t)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, T, cmap='plasma')
    ax.set_title('Решение уравнения теплопроводности')

    azim = -135
    ax.view_init(elev=20, azim=azim)

    ax.set_xlabel('x, м')
    ax.set_ylabel('t, с')
    ax.set_zlabel('T, К')

    if save:
        if folder is not None:
            if not os.path.isdir(folder):
                os.mkdir(folder)

            plt.savefig(f'{folder}/Surface.png')
        else:
            print('Укажи папку для сохранения графиков!')
            exit()

    plt.show()


def initial_conditions(x: np.array, t: np.array) -> np.array:
    """
    Применяет все начальные и граничные условия
    :param x: массив с координатами
    :param t: массив со временем
    :return: массив T
    """
    T = np.zeros((t.size, x.size))

    # Добавляем начальные условия
    T[0, :] = T_0 * (x - x_0) ** 2 * np.exp(-(x - x_0) ** 2)

    # Добавляем граничные условия
    T[:, 0] = 0
    T[:, -1] = 0

    return T


def explicit_scheme(dx: float, dt: float, save: bool = False):
    """
    Решает уравнение явной схемой
    :param dx: шаг по x
    :param dt: шаг по t
    :param save: Если True, сохраняет графики на диск. Если False, только показывает
    :return: None
    """

    if save:
        folder = 'Explicit Scheme'
        if folder is not None:
            if not os.path.isdir(folder):
                os.mkdir(folder)
        folder = f'{folder}/dx = {dx:.3f}, dt = {dt:.3f}'
    else:
        folder = ''

    x = np.arange(0, x_max + dx, step=dx)
    t = np.arange(0, t_max + dt, step=dt)
    T = initial_conditions(x=x, t=t)

    plot_graphs(T[0], x, t[0], save=save, folder=folder)

    pbar = tqdm(total=t[:-1].size)

    for n in np.arange(t.size - 1):
        for j in np.arange(1, x.size - 2):
            T[n + 1, j] = T[n, j] + dt / dx ** 2 * (T[n, j + 1] - 2 * T[n, j] + T[n, j - 1])

        # Добавляем граничные условия
        T[:, 0] = 0
        T[:, -1] = 0

        pbar.set_description(f'Явная схема. Посчитано для t = {t[n + 1]:.3f}')
        pbar.update(1)

        plot_graphs(T[n + 1], x, t[n + 1], save=save, folder=folder)

    surface_graph(T, x, t, save=True, folder=folder)
    if save:
        save_gif(folder=folder)


def crank_nicolson_method(dx: float, dt: float, save: bool = False):
    """
    Решает уравнение методом Кранка-Николсона
    :param dx: шаг по x
    :param dt: шаг по t
    :param save: Если True, сохраняет графики на диск. Если False, только показывает
    :return: None
    """

    if save:
        folder = 'Crank Nicolson Method'
        if folder is not None:
            if not os.path.isdir(folder):
                os.mkdir(folder)

        folder = f'{folder}/dx = {dx:.3f}, dt = {dt:.3f}'
    else:
        folder = ''

    x = np.arange(0, x_max + dx, step=dx)
    t = np.arange(0, t_max + 2 * dt, step=dt)

    alpha = dt / dx ** 2

    a = 0 * x
    a[-1] = -1

    b = 0 * x
    b[-1] = 2 * (1 / alpha + 1)

    f = 0 * x

    T = initial_conditions(x, t)

    plot_graphs(T[0], x, t[0], save=save, folder=folder)

    pbar = tqdm(total=t.size - 1)

    for n in np.arange(t.size - 1):
        for j in np.arange(x.size - 2, 1, -1):
            a[j - 1] = 1 / (2 * (1 + 1 / alpha) - a[j])
            f[j] = T[n, j + 1] + 2 * (1 / alpha - 1) * T[n, j] + T[n, j - 1]
            b[j - 1] = (f[j] + b[j]) / (2 * (1 / alpha + 1) - a[j])

        for j in np.arange(1, x.size - 1):
            T[n + 1, j + 1] = a[j] * T[n + 1, j] + b[j]

        pbar.set_description(f'Cхема Кранка-Николсона. Посчитано для t = {t[n]:.3f}')
        pbar.update(1)

        plot_graphs(T[n], x, t[n], save=save, folder=folder)

    surface_graph(T, x, t, save=save, folder=folder)
    if save:
        save_gif(folder=folder)


def diffusion(dx: float, dt: float, save: bool = False):
    """
    Строит графики сеточной диффузии
    :param save: Если True, сохраняет график в корневую папку
    :param dx: шаг по x
    :param dt: шаг по t
    :return: None
    """
    f = np.arange(0, t_max + dt, .001)
    alpha = dt / dx ** 2
    G_1 = -np.log(np.abs(1 - 4 * alpha * np.sin(f * np.pi / 2)**2))
    G_2 = -np.log(np.abs(1 - 2 * alpha * np.sin(f * np.pi / 2)**2)) / (1 + 2 * alpha * np.sin(f * np.pi / 2)**2)
    G_3 = 4 * dt * (f / dx) ** 2

    plt.plot(f, G_1, label='Явная схема')
    plt.plot(f, G_2, label='Cхема Кранка-Николсона')
    plt.plot(f, G_3, label='Исходное уравнение')

    plt.ylim(-.5, 3)
    plt.xlim(0, 1)

    plt.ylabel(r'$\frac{\Gamma\Delta t}{\pi}$')
    plt.xlabel(r'$\frac{\kappa}{\kappa_N}$')

    plt.grid()
    plt.legend()

    if save:
        plt.savefig(f'Diffusion dx = {dx:.1f} dt = {dt:.3f}.png')

    plt.show()


dx = .08
dt = .01

explicit_scheme(dx=dx, dt=dt, save=True)
# crank_nicolson_method(dx=dx, dt=dt, save=True)
# diffusion(dx, dt, save=True)
