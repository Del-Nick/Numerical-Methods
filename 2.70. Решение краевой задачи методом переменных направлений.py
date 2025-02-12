import numpy as np
from numpy import exp, sin, cos, pi
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from numba import njit

Nx, Ny, Nt = 100, 100, 100  # количество шагов
X1, X2 = 0, pi  # граничные условия
Y1, Y2 = 0, pi
T = 5  # время наблюдения

x = np.linspace(X1, X2, Nx)
y = np.linspace(Y1, Y2, Ny)
t = np.linspace(0, T, Nt)

h_x = pi / Nx
h_y = pi / Ny
tau = T / Nt

gamma_x = tau / (h_x ** 2)
gamma_y = tau / (h_y ** 2)

saving = True

u = np.zeros((Nx, Ny, Nt * 2 + 1))

# начальное условие
for i in range(0, Nx):
    for j in range(0, Ny):
        u[i, j, 0] = sin(2 * x[i]) * cos(y[j])


@njit
def F_1(i1, i2, j, u_new):
    return 0.5 * gamma_y * (u_new[i1][i2 - 1][j - 1] + u_new[i1][i2 + 1][j - 1]) + (1 - gamma_y) * u_new[i1][i2][j - 1] \
           + 0.5 * tau * exp(tau * (j + 1) / 2) * sin(x[i1]) * cos(y[i2])


@njit
def F_2(i1, i2, j, u_new):
    return 0.5 * gamma_x * (u_new[i1 - 1][i2][j - 1] + u_new[i1 + 1][i2][j - 1]) + (1 - gamma_x) * u_new[i1][i2][j - 1] \
           + 0.5 * tau * exp(tau * (j - 1) / 2) * sin(x[i1]) * cos(y[i2])


@njit
def progonka_x(i2, j, u_new):
    d = np.zeros(Nx)
    sigma = np.zeros(Nx)
    d[1] = 0  # условие Неймана - 1, условие Дирихле - 0
    sigma[1] = 0

    A = 0.5 * gamma_x
    B = 1 + gamma_x
    C = 0.5 * gamma_x

    u_new[0, i2, j] = 0
    for m in range(1, Nx - 1):
        Fm = -F_1(m, i2, j, u_new)
        d[m + 1] = C / (B - A * d[m])
        sigma[m + 1] = (Fm - A * sigma[m]) / (A * d[m] - B)
        u_new[Nx - 1][i2][j] = sigma[-1] / (1 - d[-1])  # условие Дирихле

    for m in range(Nx - 1, 0, -1):
        u_new[m - 1][i2][j] = d[m] * u_new[m][i2][j] + sigma[m]

    return u_new


@njit
def progonka_y(i1, j, u_new):
    d = np.zeros(Ny)
    sigma = np.zeros(Ny)
    d[1] = 1  # условие Неймана - 1, условие Дирихле - 0
    sigma[1] = 0

    A = 0.5 * gamma_y
    B = 1 + gamma_y
    C = 0.5 * gamma_y

    for m in range(1, Ny - 1):
        Fm = -F_2(i1, m, j, u_new)
        d[m + 1] = C / (B - A * d[m])
        sigma[m + 1] = (Fm - A * sigma[m]) / (A * d[m] - B)
    u_new[i1][Ny - 1][j] = sigma[-1] / (1 - d[-1])
    for m in range(Ny - 1, 0, -1):
        u_new[i1][m - 1][j] = d[m] * u_new[i1][m][j] + sigma[m]

    return u_new


for j in tqdm(range(1, 2 * Nt, 2)):
    for i2 in range(1, Ny - 1):
        u = progonka_x(i2, j, u)
    for i1 in range(1, Nx - 1):
        u = progonka_y(i1, j+1, u)
    # for k in range(Ny):
    #     u[0, k, j+1] = 0
    #     u[Nx-1, k, j + 1] = 0



# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ

z = np.zeros((Nx, Ny, Nt * 2 + 1))

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nt):
            z[i, j, k] = exp(-5 * k*tau) * sin(2 * x[i]) * cos(y[j]) + (exp(k*tau) - exp(-2 * k*tau)) / 3 * sin(x[i]) * cos(y[j])

print(np.max(u[:, :, :]), np.max(z[:, :, :]))
print(z.shape, u.shape)

# x_grid, t_grid = np.meshgrid(x[1:-2], y[1:-2])
# fig = plt.figure(figsize=(8, 6))
# ax_3d = plt.subplot(projection='3d')
# ax_3d.plot_surface(x_grid, t_grid, u[1:-2, 1:-2, 0].T-z[1:-2, 1:-2, 0].T, rstride=5, cstride=5, cmap='plasma')
# # ax_3d.plot_surface(x_grid, t_grid, u_null[1:-2, 1:-2, 0], rstride=5, cstride=5, cmap='viridis')
# ax_3d.set_xlabel('x')
# ax_3d.set_ylabel('y')
# ax_3d.set_zlabel('U')
# plt.title(f'Численное решение z(x, y, t) в момент времени t = 0')
# plt.show()

# plt.ion()
# fig = plt.figure(figsize=(16, 6))
# for timer in tqdm(range(len(t)), desc='График аналитического решения'):
#     plt.clf()
#     plt.subplot(1, 2, 1)
#     plt.pcolormesh(x, y, z[:, :, timer], cmap='inferno')
#     plt.title(f'Аналитическоге решение z(x, y, t) в момент времени t = {t[timer]:.1f}')
#     plt.colorbar()
#     plt.xlabel('Координата по х', fontsize=12)
#     plt.ylabel('Координата по y', fontsize=12)
#     plt.subplot(1, 2, 2)
#     plt.pcolormesh(x, y, u[:, :, timer*2], cmap='inferno')
#     plt.colorbar()
#     plt.xlabel('Координата по х', fontsize=12)
#     plt.ylabel('Координата по y', fontsize=12)
#     plt.title(f'Численное решение z(x, y, t) в момент времени t = {t[timer]:.1f}')
#     plt.draw()
#     plt.gcf().canvas.flush_events()


# plt.ion()
# fig = plt.figure(figsize=(8, 6))
# for timer in tqdm(range(len(t)), desc='График численного решения'):
#     plt.clf()
#     plt.pcolormesh(y, x, u[:, :, timer], cmap='inferno')
#     plt.colorbar()
#     plt.ylabel('Координата по х', fontsize=12)
#     plt.xlabel('Координата по y', fontsize=12)
#     plt.title(f'Численное решение z(x, y, t) в момент времени t = {t[timer]:.1f}')
#     plt.draw()
#     plt.gcf().canvas.flush_events()

for timer in range(Nt):
    x_grid, y_grid = np.meshgrid(x[1:-2], y[1:-2])
    ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, y_grid, z[1:-2, 1:-2, timer].T, cmap='plasma')
    # ax_3d.plot_surface(x_grid, y_grid, z[1:-2:, 1:-2, timer], cmap='viridis')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('U')
    # plt.title(f'Численное решение z(x, y, t) в момент времени t = 1')
    # plt.show()


    # ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, y_grid, u[1:-2, 1:-2, timer].T, cmap='viridis')
    # ax_3d.set_xlabel('x')
    # ax_3d.set_ylabel('y')
    # ax_3d.set_zlabel('U')
    plt.title(f'Аналитическое решение в начальный момент времени t = {timer*tau:.2f}')
    plt.show()

# x_grid, y_grid = np.meshgrid(x[1:-2], y[1:-2])
# plt.ion()
# fig = plt.figure(figsize=(8, 6))
# for timer in tqdm(range(len(t)), desc='График численного решения'):
#     plt.clf()
#     ax_3d = plt.subplot(projection='3d')
#     ax_3d.plot_surface(x_grid, y_grid, u[1:-2, 1:-2, timer*2], cmap='plasma')
#     # ax_3d.plot_surface(x_grid, y_grid, z[1:-2:, 1:-2, timer], cmap='viridis')
#     ax_3d.set_xlabel('x')
#     ax_3d.set_ylabel('y')
#     ax_3d.set_zlabel('U')
#     # ax_3d.set_zlim(-30, 30)
#     # ax_3d.view_init(elev=0 + timer / 4, azim=timer)
#     # ax_3d.dist = 6 + timer / 50
#     ax_3d.view_init(elev=0, azim=90)
#     ax_3d.dist = 7
#     plt.title(f'Численное решение z(x, y, t) в момент времени t = {t[timer]:.1f}')
#     plt.draw()
#     plt.gcf().canvas.flush_events()
#
# x_grid, t_grid = np.meshgrid(x, y)
# plt.ion()
# fig = plt.figure(figsize=(8, 6))
# for timer in tqdm(range(len(t)), desc='График аналитического решения'):
#     plt.clf()
#     ax_3d = plt.subplot(projection='3d')
#     ax_3d.plot_surface(x_grid, t_grid, z[:, :, timer], rstride=5, cstride=5, cmap='plasma')
#     ax_3d.set_xlabel('x')
#     ax_3d.set_ylabel('t')
#     ax_3d.set_zlabel('U')
#     # ax_3d.set_zlim(-30, 30)
#     ax_3d.view_init(elev=0 + timer / 4, azim=timer)
#     ax_3d.dist = 6 + timer / 50
#     plt.title(f'Аналитическое решение z(x, y, t) в момент времени t = {t[timer]:.1f}')
#     plt.draw()
#     plt.gcf().canvas.flush_events()


x_grid, t_grid = np.meshgrid(x[1:-2], y[1:-2])
plt.ion()
fig = plt.figure(figsize=(8, 6))
for timer in tqdm(range(len(t)), desc='График относительной ошибки'):
    plt.clf()
    ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, t_grid, z[1:-2, 1:-2, timer].T-u[1:-2, 1:-2, timer].T, rstride=5, cstride=5, cmap='plasma')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('U')
    # ax_3d.set_zlim(-30, 30)
    ax_3d.view_init(elev=0 + timer / 4, azim=timer)
    ax_3d.dist = 6 + timer / 50
    plt.title(f'Относительная ошибка в момент времени t = {t[timer]:.1f}')
    plt.draw()
    plt.gcf().canvas.flush_events()


x_grid, t_grid = np.meshgrid(x[1:-2], y[1:-2])
plt.ion()
fig = plt.figure(figsize=(8, 6))
for timer in tqdm(range(len(t)), desc='График относительной ошибки'):
    plt.clf()
    ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, t_grid, u[1:-2, 1:-2, timer].T, rstride=5, cstride=5, cmap='plasma')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('U')
    # ax_3d.set_zlim(-30, 30)
    ax_3d.view_init(elev=0 + timer / 4, azim=timer)
    ax_3d.dist = 6 + timer / 50
    plt.title(f'Относительная ошибка в момент времени t = {t[timer]:.1f}')
    plt.draw()
    plt.gcf().canvas.flush_events()
    if saving:
