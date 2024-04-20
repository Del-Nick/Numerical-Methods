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






#________________________________________________________________________
# from numpy import empty, zeros, linspace, meshgrid, cos, sin, exp, log, sqrt, sum, pi
# from matplotlib.pyplot import figure, subplots, xlim, ylim
# from mpl_toolkits.mplot3d import Axes3D
# from IPython import get_ipython
# # get_ipython().run_line_magic('matplotlib', 'qt')
# # from celluloid import Camera
# import matplotlib.pyplot as plt
#
#
# def progonka(A, B, C, F, kappa, mu, N):
#     # -------------------------------------------------------------------------------------------
#     # Вид системы:
#
#     # -y_{0} + kappa[0]*y_{1}= -mu[0];
#
#     # A*y_{n-1} - C*y_{n} + B*y_{n+1} = -F_{n}, n=1,2,...,N-1;
#
#     # kappa[1]*y_{N-1} - y_{N} = -mu[1].
#
#     # -------------------------------------------------------------------------------------------
#     # Используемые обозначения:
#
#     # kappa[0]=B_0/C_0; kappa[1]=A_N/C_N;
#
#     # mu[0]=F_{0}/C_0; mu[1]=F_{N}/C_N.
#
#     # -------------------------------------------------------------------------------------------
#     # Для аппроксимации со вторым порядком на левой границе:
#     # kappa[0] = (4*B - C)/(3*B - A); mu[0] = (F[1] - 2*h*B*alpha)/(3*B - A); u'(left) = alpha
#
#     # Для аппроксимации со вторым порядком на правой границе:
#     # kappa[1] = (C - 4*A)/(B - 3*A); mu[1] = -(F[N-1] + 2*h*A*beta)/(B - 3*A); u'(right) = beta
#
#     # -------------------------------------------------------------------------------------------
#     # Создание столбца из N нулей, в который будет записано решение:
#     Y = zeros(N + 1)
#     # -------------------------------------------------------------------------------------------
#     # Создание массивов прогоночных коэффициентов:
#     alpha = zeros(N)
#     beta = zeros(N)
#
#     alpha[0] = kappa[0]
#     beta[0] = mu[0]
#     # -------------------------------------------------------------------------------------------
#     # Прямой ход прогонки:
#     for n in range(N - 1):
#         alpha[n + 1] = B / (C - A * alpha[n])
#         beta[n + 1] = (A * beta[n] + F[n + 1]) / (C - A * alpha[n])
#     # -------------------------------------------------------------------------------------------
#     # Обратный ход прогонки:
#     Y[N] = (mu[1] + kappa[1] * beta[N - 1]) / (1 - kappa[1] * alpha[N - 1])
#
#     for n in range(N - 1, -1, -1):
#         Y[n] = alpha[n] * Y[n + 1] + beta[n]
#     # -------------------------------------------------------------------------------------------
#
#     return Y
#
#
# def Num_Sol(nodes, init_points, int_length, r_coef=[2, 2, 2], s_coef=[0, 0, 0]):
#     # -------------------------------------------------------------------------------------------
#     # ВХОДНЫЕ ПАРАМЕТРЫ:
#
#     # nodes = [N_0, M_0, J_0]
#     # N_0 - число интервалов базовой сетки по координате x
#     # M_0 - число интервалов базовой сетки по координате y
#     # J_0 - число интервалов базовой сетки по времени
#
#     # init_points = [x_0, y_0, t_0]
#     # x_0 - начало отсчета по координате x
#     # y_0 - начало отсчета по координате y
#     # t_0 - начало отсчета по времени
#
#     # int_length = [Lx, Ly, T]
#     # Lx - длина интервала по координате x
#     # Ly - длина интервала по координате y
#     # T - длина интервала по времени
#
#     # r_coef = [rx, ry, rt]
#     # rx, ry, rt - коэффициенты сгущения сетки
#
#     # s_coef = [sx, sy, st]
#     # (sx, sy, st) - номер сетки, на которой вычисляется решение
#     # (если sx = sy = st = 0, то решение ищется на базовой сетке)
#     # -------------------------------------------------------------------------------------------
#     # ВВЕДЕНИЕ СЕТКИ:
#
#     # Распаковка значений
#     N_0 = nodes[0];
#     M_0 = nodes[1];
#     J_0 = nodes[2]
#     x_0 = init_points[0];
#     y_0 = init_points[1];
#     t_0 = init_points[2]
#     Lx = int_length[0];
#     Ly = int_length[1];
#     T = int_length[2]
#     rx = r_coef[0];
#     ry = r_coef[1];
#     rt = r_coef[2]
#     sx = s_coef[0];
#     sy = s_coef[1];
#     st = s_coef[2]
#
#     # Вычисление числа интервалов на сетке с номером (sx, sy, st)
#     N = N_0 * rx ** sx
#     M = M_0 * ry ** sy
#     J = J_0 * rt ** st
#
#     hx = Lx / N  # Шаг сетки по координате x
#     hy = Ly / M  # Шаг сетки по координате y
#     tau = T / J  # Шаг сетки по времени
#
#     x = linspace(x_0, x_0 + Lx, N + 1)  # Сетка по координате x
#     y = linspace(y_0, y_0 + Ly, M + 1)  # Сетка по координате y
#     t = linspace(t_0, t_0 + T, J + 1)  # Сетка по времени
#     # -------------------------------------------------------------------------------------------
#     # ВЫДЕЛЕНИЕ ПАМЯТИ ПОД МАССИВ СЕТОЧНЫХ ЗНАЧЕНИЙ РЕШЕНИЯ:
#
#     u = empty((N + 1, M + 1, J + 1))  # на сгущенной сетке
#     u_basic = empty((N_0 + 1, M_0 + 1, J_0 + 1))  # на базовой сетке
#
#     # Начальное условие...................................................(Менять под свою задачу)
#     for n in range(N + 1):
#         for m in range(M + 1):
#             u[n, m, 0] = sin(2 * x[n]) * cos(y[m])
#     # -------------------------------------------------------------------------------------------
#     # РЕШЕНИЕ УРАВНЕНИЯ НА ВНУТРЕННИХ ТОЧКАХ:
#
#     # На каждой итерации внешнего цикла совершается переход на новый временной слой
#     for j in range(J):
#
#         # -------------------------------------------------------------------------------------------
#         # ЗАПОЛНЕНИЕ ВСПОМОГАТЕЛЬНОГО МАССИВА u_05:
#
#         u_05 = empty((N + 1, M + 1))
#         kappa = empty(2)
#         mu = empty(2)
#
#         A = tau / (2 * hx ** 2)
#         C = (1 + tau / hx ** 2)
#         B = tau / (2 * hx ** 2)
#
#         kappa[0] = 0
#         kappa[1] = 0
#         mu[0] = 0
#         mu[1] = 0
#
#         for m in range(1, M):
#             for n in range(1, N):
#                 F = u[n, m, j] + tau / (2 * hy ** 2) * (u[n, m + 1, j] - 2 * u[n, m, j] + u[n, m - 1, j]) + 0.5 * tau * exp(tau * (j + 1) / 2) * sin(x[n]) * cos(y[m])
#
#             u_05[:, m] = progonka(A, B, C, F, kappa, mu, N)
#         # -------------------------------------------------------------------------------------------
#         # ПЕРЕХОД НА СЛОЙ j+1:
#
#         A = tau / (2 * hy ** 2)
#         C = (1 + tau / hy ** 2)
#         B = tau / (2 * hy ** 2)
#
#         kappa[0] = 0
#         kappa[1] = (C - 4 * A) / (B - 3 * A)
#
#         for n in range(1, N):
#             for m in range(1, M):
#                 F = u_05[n, m] + tau / (2 * hx ** 2) * (u_05[n + 1, m] - 2 * u_05[n, m] + u_05[n - 1, m]) + 0.5 * tau * exp(tau * (j - 1) / 2) * sin(x[n]) * cos(y[m])
#
#                 mu[0] = sin(x[n])
#                 mu[1] = -(F[M - 1] + 2 * hy * A * sin(2 * x[n])) / (B - 3 * A)
#
#                 u[n, m, j + 1] = progonka(A, B, C, F, kappa, mu, M)
#
#         # Завершение перехода на слой j+1 через граничные условия
#         u[0, :, j + 1] = zeros(M + 1)
#         u[N, :, j + 1] = zeros(M + 1)
#         # -------------------------------------------------------------------------------------------
#
#     # -------------------------------------------------------------------------------------------
#     # ИЗ МАССИВА u ВЫБРАЮТСЯ СЕТОЧНЫЕ ЗНАЧЕНИЯ ИЗ УЗЛОВИ,
#     # СОВПАДАЮЩИХ С УЗЛАМИ БАЗОВОЙ СЕТКИ:
#
#     u_basic = u[::rx ** sx, ::ry ** sy, ::rt ** st]
#     # -------------------------------------------------------------------------------------------
#
#     return u, u_basic, x, y, t
#
#
# def PrintTriangular(A,i) :
#     print('    ',end=' ')
#     for l in range(len(A)) :
#         print(' p={0:<2d}'.format(p + l*q),end=' ')
#     print()
#     for m in range(len(A)) :
#         print('s={0:<2d}'.format(m),end=' ')
#         for l in range(m + 1 - i) :
#             print('{0:5.3f}'.format(A[m,l]),end=' ')
#         print()
#     print()
#
#
# #-------------------------------------------------------------------------------------------
# # ВХОДНЫЕ ПАРАМЕТРЫ:
#
# N = 200
# M = 200
# J = 200
# nodes = [N, M, J]
#
# x_0 = 0.
# y_0 = 0.
# t_0 = 0.
# init_points = [x_0, y_0, t_0]
#
# Lx = pi
# Ly = pi
# T = 1.
# int_length = [Lx, Ly, T]
#
# #-------------------------------------------------------------------------------------------
# # ЧИСЛЕНННОЕ РЕШЕНИЕ:
#
# u, u_basic, x, y, t = Num_Sol(nodes, init_points, int_length)
#
#
# # 2D ГРАФИКИ:
#
# times = [0.25, 0.4, 0.5,0.75, 0.8, 1]
# fig = plt.figure(figsize=(18,9))
# for i in range(len(times)):
#     plt.subplot(2,3,i+1)
#     r = int(times[i] / T * (J))
#     plt.pcolormesh(y,x,u[:,:,r], cmap='coolwarm', vmin = 0, vmax = 1.7)
#     plt.colorbar()
#     plt.ylabel('Координата x', fontsize=12)
#     plt.xlabel('Координата y', fontsize=12)
#     plt.title('Значение функции в момент времени t = '+str(times[i]), fontsize=12)
#     plt.tight_layout()
#
# plt.show()