import multiprocessing

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from matplotlib.colors import Normalize
from numba import njit
from numpy import arange, linspace, ndarray, zeros, meshgrid, concatenate
from numpy import exp, pi, abs, max
from numpy.fft import fftshift
from tqdm import tqdm

"""
 ██████ ██████  ██    ██      ██████  ██████  ██    ██ ███    ██ ████████ 
██      ██   ██ ██    ██     ██      ██    ██ ██    ██ ████   ██    ██    
██      ██████  ██    ██     ██      ██    ██ ██    ██ ██ ██  ██    ██    
██      ██      ██    ██     ██      ██    ██ ██    ██ ██  ██ ██    ██    
 ██████ ██       ██████       ██████  ██████   ██████  ██   ████    ██    
                                                                                                                                              
Определяем число ядре в процессоре и настраиваем бэкенд для распараллеливания
"""
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
print(f'В процессоре {multiprocessing.cpu_count()} ядер')

"""                                                 
███    ███  █████  ██ ███    ██     ██████   █████  ██████   █████  ███    ███ ███████ ████████ ███████ ██████  ███████ 
████  ████ ██   ██ ██ ████   ██     ██   ██ ██   ██ ██   ██ ██   ██ ████  ████ ██         ██    ██      ██   ██ ██      
██ ████ ██ ███████ ██ ██ ██  ██     ██████  ███████ ██████  ███████ ██ ████ ██ █████      ██    █████   ██████  ███████ 
██  ██  ██ ██   ██ ██ ██  ██ ██     ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██         ██    ██      ██   ██      ██ 
██      ██ ██   ██ ██ ██   ████     ██      ██   ██ ██   ██ ██   ██ ██      ██ ███████    ██    ███████ ██   ██ ███████                                                                                                               
"""

x_min, x_max = -5, 5
y_min, y_max = -5, 5
z_min, z_max = 0, 0.4

Rv = -11

Nx, Ny = 1024, 1024

ax_x = linspace(x_min, x_max, 1024, dtype='float64')
ax_y = linspace(y_min, y_max, 1024, dtype='float64')
ax_z = arange(z_min, z_max + .01, .01, dtype='float64')

dx, dy, dz = ax_x[1] - ax_x[0], ax_y[1] - ax_y[0], ax_z[1] - ax_z[0]


def fft(a: ndarray) -> ndarray:
    """
    Распараллеленный на все ядра процессора вариант Фурье-преобразования.
    Примерно в 3 раза быстрее, чем Фурье-преобразование от NumPy.
    :param a: двумерный массив numpy
    :return: двумерный массив с преобразованием
    """
    get_fft2_array = pyfftw.builders.fft2(a, overwrite_input=True, planner_effort='FFTW_ESTIMATE',
                                          threads=multiprocessing.cpu_count(), norm='forward')
    return get_fft2_array(a)


def ifft(a: ndarray) -> ndarray:
    """
    Распараллеленный на все ядра процессора вариант обратного Фурье-преобразования. Примерно в 3 раза быстрее,
    чем Фурье-преобразование от NumPy.
    :param a: двумерный массив numpy
    :return: двумерный массив с преобразованием
    """
    get_ifft2_array = pyfftw.builders.ifft2(a, overwrite_input=True, planner_effort='FFTW_ESTIMATE',
                                            threads=multiprocessing.cpu_count(), norm='forward')
    return get_ifft2_array(a)


def get_3D_graph(E: ndarray, show: bool = True, save: bool = False, folder: str = None):
    """
    Строит поверхность интенсивности в точках z=0, z=z_max, y = (y_max - y_min) / 2
    :param E: поле, трёхмерный массив
    :param show: вкл/выкл отображения графиков
    :param save: вкл/выкл сохранения графиков
    :param folder: папка для сохранения графиков
    """
    X, Y = meshgrid(ax_x, ax_y)

    for z_index in [0, ax_z.size - 1]:
        fig, ax = plt.subplots(num='Ветровая рефракция пучка', figsize=(8, 8), subplot_kw={"projection": "3d"})
        ax.set_title(f'z = {ax_z[z_index] * 100:.0f} см')
        ax.plot_surface(X, Y, abs(E[z_index]) ** 2, cmap='viridis')

        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')
        ax.set_zlabel('Интенсивность')

        plt.tight_layout()

        ax.dist = 5
        ax.azim = 45
        ax.elev = 30

        ax.set_zlim(0, max(abs(E) ** 2))

        if save:
            if folder:
                fig.savefig(f'{folder}/3D Wind refraction at z = {ax_z[z_index]:.2f} with Rv = {Rv}.png', dpi=720)
            else:
                fig.savefig(f'3D Wind refraction at z = {ax_z[z_index]:.2f} with Rv = {Rv}.png', dpi=720)

        if show:
            plt.show()

        plt.close(fig)

    X, Z = meshgrid(ax_x, ax_z)
    fig, ax = plt.subplots(num='Ветровая рефракция пучка от z', figsize=(8, 8), subplot_kw={"projection": "3d"})
    ax.set_title(f'y = {ax_y[ax_y.size // 2]:.0f} м')
    ax.plot_surface(X, Z, abs(E[:, ax_x.size // 2, :]) ** 2, cmap='viridis', linewidth=0)

    plt.tight_layout()
    ax.dist = 5
    ax.azim = 45
    ax.elev = 30

    ax.set_xlabel('X, м')
    ax.set_ylabel('Z, м')
    ax.set_zlabel('Интенсивность')

    ax.set_zlim(0, max(abs(E) ** 2))

    if save:
        if folder:
            fig.savefig(f'{folder}/3D Wind refraction at y = {ax_y[ax_y.size // 2]:.2f} with Rv = {Rv}.png', dpi=720)
        else:
            fig.savefig(f'3D Wind refraction at y = {ax_y[ax_y.size // 2]:.2f} with Rv = {Rv}.png', dpi=720)

    if show:
        plt.show()

    plt.close(fig)


def get_2D_graph(E: ndarray, show: bool = True, save: bool = False, folder: str = None):
    """
    Строит 2D цветовую карту интенсивности в точках z=0, z=z_max, y = (y_max - y_min) / 2
    :param E: поле, трёхмерный массив
    :param show: вкл/выкл отображения графиков
    :param save: вкл/выкл сохранения графиков
    :param folder: папка для сохранения графиков
    """
    E = abs(E) ** 2

    fig, axes = plt.subplots(nrows=1, ncols=2, num='Ветровая рефракция пучка', figsize=(10, 4))

    normalizer = Normalize(0, max(E))
    im = cm.ScalarMappable(norm=normalizer)

    plt.subplot(1, 2, 1).set_title(f'z = 0 см')
    plt.pcolormesh(ax_x, ax_y, E[0], cmap='viridis', norm=normalizer)

    plt.xlabel('X, м')
    plt.ylabel('Y, м')

    plt.subplot(1, 2, 2).set_title(f'z = {ax_z[-1]} см')
    plt.pcolormesh(ax_x, ax_y, E[-1], cmap='viridis', norm=normalizer)

    plt.xlabel('X, м')
    plt.ylabel('Y, м')

    fig.colorbar(im, ax=axes.ravel().tolist())

    if save:
        if folder:
            fig.savefig(f'{folder}/2D Wind refraction at z with Rv = {Rv}.png', bbox_inches='tight', dpi=720)
        else:
            fig.savefig(f'2D Wind refraction at z with Rv = {Rv}.png', bbox_inches='tight', dpi=720)

    if show:
        plt.show()

    plt.close()

    plt.figure(num='Ветровая рефракция пучка от z', figsize=(8, 8))
    plt.pcolormesh(ax_z, ax_x, E[:, ax_y.size // 2, :].T, cmap='viridis')
    plt.title(f'y = {ax_y[ax_y.size // 2]:.0f} м')
    plt.colorbar(ticks=linspace(0, max(abs(E) ** 2), 5))

    plt.xlabel('Z, м')
    plt.ylabel('X, м')

    plt.tight_layout(pad=0.1)

    if save:
        if folder:
            plt.savefig(f'{folder}/2D Wind refraction at y = {ax_y[ax_y.size // 2]:.2f} with Rv = {Rv}.png', dpi=720)
        else:
            plt.savefig(f'2D Wind refraction at y = {ax_y[ax_y.size // 2]:.2f} with Rv = {Rv}.png', dpi=720)

    if show:
        plt.show()


def get_plot(E: ndarray, show: bool = True, save: bool = False, folder: str = None):
    """
    Строит график максимальной интенсивности для каждой плоскости XY от Z
    :param E: поле, трёхмерный массив NumPy
    :param show: вкл/выкл отображения графиков
    :param save: вкл/выкл сохранения графиков
    :param folder: папка для сохранения графиков
    """
    E = abs(E) ** 2

    plt.figure(num='Максимальная интенсивность от z', figsize=(8, 8))
    plt.plot(ax_z, [max(E[i, :, ax_y.size // 2]) for i in range(ax_z.size)])
    plt.grid()
    plt.title('Зависимость максимальной интенсивности от z')

    plt.xlabel('X, м')
    plt.ylabel('Max intensity')

    plt.tight_layout(pad=0.01)

    if save:
        if folder:
            plt.savefig(f'{folder}/Max intensity with Rv = {Rv}.png', dpi=720)
        else:
            plt.savefig(f'Max intensity with Rv = {Rv}.png', dpi=720)

    if show:
        plt.show()


def get_grid_for_the_run_through_method() -> tuple[ndarray, ndarray]:
    i = arange(ax_x.size // 2)
    i = concatenate([i, i[::-1]])
    j = arange(ax_y.size // 2)
    j = concatenate([j, j[::-1]])
    i, j = meshgrid(i, j)
    return i, j


"""
Для ускорения расчётов создаём двумерные сетки для x, y
"""
i_ax, j_ax = get_grid_for_the_run_through_method()


@njit()
def get_linear_part_of_field(spectra: ndarray) -> ndarray:
    length = x_max - x_min
    return spectra * exp(.5j * (2 * pi / length) ** 2 * (i_ax ** 2 + j_ax ** 2) * dz)


@njit()
def get_nonlinear_part_of_field(field: ndarray) -> tuple[ndarray, ndarray]:
    T = zeros((Ny, Nx), dtype='complex128')
    I = abs(field) ** 2
    for x in np.arange(Nx - 1):
        T[:, x + 1] = T[:, x] + dx * I[:, x]
    return field * exp(-.5j * Rv * T * dz), T


x_2D, y_2D = meshgrid(ax_x, ax_y)
E = zeros(shape=(ax_z.size, ax_x.size, ax_y.size), dtype='complex128')

# Начальноре условие
E[0] = exp(-(x_2D ** 2 + y_2D ** 2) / 2)

for i, z in enumerate(tqdm(ax_z[1:], desc='Расчёт ветровой рефракции пучка'), start=1):
    E[i] = ifft(get_linear_part_of_field(spectra=fft(E[i - 1])))
    E[i], T = get_nonlinear_part_of_field(field=E[i])


show = False
save = True

get_3D_graph(E, show=show, save=save)
get_2D_graph(E, show=show, save=save)
get_plot(E, show=show, save=save)