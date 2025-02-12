import numpy as np
import matplotlib.pyplot as plt
from numpy import arctan, sin, pi, log, sqrt
from numba import njit
from tqdm import tqdm
import time

# Задаем начальные условия
L = 1  # Длина сетки
T = 2.5  # Интервал времени
Nx = 800
Nt = 4000
h = L/Nx  # Шаг по пространству
tau = T/Nt  # Шаг по времени

eps = 1e-8


def draw_characteristics():
    for x_0 in np.arange(0, 1.1, 0.1):
        x_1 = x_0 + arctan(-sin(pi * x_0)) * T
        plt.plot(T, x_1, color='green', label='$t_0 = 0$')

    for t_0 in range(5):
        x_2 = (T - t_0) * arctan(0)
        plt.plot(T, x_2, color='red', label='$x_0 = 0$')

    plt.ylim(-0.01, 1)
    plt.xlim(0, max(T))
    plt.grid()
    plt.xlabel('t, с')
    plt.ylabel('x')
    plt.title('Семейство характеристик')
    plt.show()

@njit
def f(x):
    return arctan(x) * x - log(sqrt(1+x*x))


X = np.linspace(0, 1, Nx)  # Разбивка рассматриваемой области по координате x
T = np.linspace(0, T, Nt)  # Разбивка интервала времени по времени

u_1 = np.zeros((Nx, Nt))  # Пустой массив для хранения значений решения
u_1[:, 0] = sin(pi*X)

u_2 = np.zeros((Nx, Nt))  # Пустой массив для проверки
u_2[:, 0] = sin(pi*X)


@njit
def four_point(y):
    for n in range(Nx-1):
        for j in range(Nt-1):
            un1 = y[n + 1][j]
            un2 = y[n + 1][j] + 2*eps
            while abs(un1 - un2) > eps:
                F = (y[n][j + 1] - y[n][j] + un2 - y[n + 1][j]) /2 / tau + (f(un2) - f(y[n][j + 1]) + f(y[n + 1][j]) - f(y[n][j])) / 2 / h
                dF = 1 / 2 / tau + np.arctan(un2) / 2 / h
                un1 = un2
                un2 = un2 - F / dF
                # print('main', n, j, x)
            y[n + 1][j + 1] = un2
    return y

@njit
def three_point(y):
    for n in range(Nx-1):
        for j in range(Nt-1):
            un1 = y[n][j]
            un2 = y[n][j] + 2 * eps
            while abs(un1 - un2) > eps:
                F = (un2 - y[n + 1][j]) / tau + (f(y[n + 1][j]) - f(y[n][j])) / h
                dF = 1 / 2 / tau + np.arctan(un2) / 2 / h
                un1 = un2
                un2 = un2 - F / dF
                # print('check', n, j, un2)
            y[n + 1][j + 1] = un2
    return y

timer = time.time()
u_1 = four_point(u_1)
print(f'Четырёхточечная схема посчтина за {time.time()-timer} секунд')

timer = time.time()
u_2 = three_point(u_2)
print(f'Трёхточечная схема посчтина за {time.time()-timer} секунд')

draw_characteristics()

plt.pcolormesh(T, -X, u_1, cmap='inferno')
plt.colorbar()
plt.title('Численное решение уравнения переноса\nЧетырёхточечная схема')
plt.xlabel('t, с')
plt.ylabel('x')
plt.show()

plt.pcolormesh(T, -X, u_2, cmap='inferno')
plt.colorbar()
plt.title('Численное решение уравнения переноса\nТрёхточечная схема')
plt.xlabel('t, с')
plt.ylabel('x')
plt.show()

print(f'\nМаксимальная ошибка составила {np.max(abs(u_1-u_2))}')
print(f'\nМинимальная ошибка составила {np.min(abs(u_1-u_2))}')
print(f'\nСредняя по модулю ошибка составила {np.mean(abs(u_1-u_2))}')

plt.pcolormesh(T, X, u_1-u_2, cmap='inferno', vmin=-0.011, vmax=0.01)
plt.colorbar()
plt.title('Численное решение уравнения переноса\nРазность четырёхточечной и трёхточечной схем')
plt.xlabel('t, с')
plt.ylabel('x')
plt.show()

plt.pcolormesh(T, X, np.log10(u_1-u_2), cmap='inferno')
plt.colorbar()
plt.title('Численное решение уравнения переноса\nРазность четырёхточечной и трёхточечной схем в логарифмическом масштабе')
plt.xlabel('t, с')
plt.ylabel('x')
plt.show()

x_grid, t_grid = np.meshgrid(-X, T)

fig = plt.figure()
ax_3d = plt.subplot(projection='3d')
ax_3d.plot_surface(x_grid, t_grid, (u_1-u_2).T, rstride=5, cstride=5, cmap='plasma')
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('t')
ax_3d.set_zlabel('ΔU')
# ax_3d.set_zlim(0.011, 0.01)
plt.title('Численное решение уравнения переноса\nРазность четырёхточечной и трёхточечной схем')
plt.show()

fig = plt.figure()
ax_3d = plt.subplot(projection='3d')
ax_3d.plot_surface(x_grid, t_grid, (u_1-u_2).T/np.max(abs(u_1-u_2)), rstride=5, cstride=5, cmap='plasma')
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('t')
ax_3d.set_zlabel('ΔU')
ax_3d.set_zlim(0, 1)
plt.title('Численное решение уравнения переноса\nНормированная разность четырёхточечной и трёхточечной схем')
plt.show()

fig = plt.figure()
ax_3d = plt.subplot(projection='3d')
ax_3d.plot_surface(x_grid, t_grid, u_1.T, rstride=5, cstride=5, cmap='plasma')
ax_3d.set_xlabel('x')
ax_3d.set_ylabel('t')
ax_3d.set_zlabel('U')
# ax_3d.set_zlim(0, 1)
plt.title('Численное решение уравнения переноса\nЧетырёхточечная схема')
plt.show()
