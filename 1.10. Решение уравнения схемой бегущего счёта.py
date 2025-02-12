import numpy as np
import matplotlib.pyplot as plt
from numpy import arctan, sin, pi, log, sqrt
from numba import njit
from tqdm import tqdm
import time
import imageio

# Задаем начальные условия
X_START, X_END = 0, -1  # Длина сетки
T_START, T_END = 0, 5  # Интервал времени
epsilon = 1e-8
Nx, Nt = 400, 1000
h = abs(X_END-X_START) / Nx  # Шаг по пространству
tau = abs(T_END-T_START) / Nt  # Шаг по времени

saving = True


def draw_characteristics():
    for x_0 in np.linspace(0, -1, 25):
        x_1 = x_0 + arctan(-sin(pi * x_0)) * T
        plt.plot(x_1, T, color='#BA5400', label='$t_0 = 0$')

    for t_0 in range(5):
        x_2 = (T - t_0) * arctan(0)
        plt.plot(x_2, T, color='#007070', label='$x_0 = 0$')

    plt.ylim(T_START, T_END)
    plt.xlim(X_END, 0.001)
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('T')
    plt.title('Семейство характеристик')
    plt.show()


X = np.linspace(X_END, X_START, Nx)  # Разбивка рассматриваемой области по координате x
T = np.linspace(T_START, T_END, Nt)  # Разбивка интервала времени по времени

# draw_characteristics()


@njit
def F(m, n, y):
    return arctan(y[m][n]) * y[m][n] - log(sqrt(1+y[m][n]*y[m][n]))


@njit
def der_f(m1, n1, y):
    return 1 / 2 / ht - arctan(y[m1][n1]) / 2 / hx


@njit
def f(mp1, np1, y):
    n = np1 - 1
    m = mp1 - 1
    return (y[mp1][n] - y[m][n] + y[mp1][np1] - y[m][np1]) / (2 * ht) - (
                F(mp1, np1, y) - F(mp1, n, y) + F(m, np1, y) - F(m, n, y)) / (2 * hx)


# Grid 's parameters
hx = (X_END - X_START) / (Nx - 1)
ht = (T_END - T_START) / (Nt - 1)
y = np.zeros((Nt, Nx))

timer = time.time()

# Conditions
y[0, :] = -sin(X * pi)  # initial


# ------------- СЧИТАЕМ ПО ЧЕТЫРЁХТОЧЕЧНОЙ СХЕМЕ -------------

for m in tqdm(np.arange(Nt)[0:Nt - 1], desc='Расчёт по четырёхточечной схеме'):
    for n in np.arange(Nx)[0:Nx - 1]:
        eps = epsilon + 1
        while eps > epsilon:
            ep = f(m + 1, n + 1, y) / der_f(m + 1, n + 1, y)
            y[m + 1][n + 1] = y[m + 1][n + 1] - ep
            eps = np.abs(ep)


print(f'Четырёхточечная схема посчтина за {time.time() - timer} секунд')

xn = np.linspace(X_START, X_END, Nx)
tm = np.linspace(T_START, T_END, Nt)

x_grid, t_grid = np.meshgrid(xn, tm)
plt.ion()
if saving:
    step = 1
else:
    step = 10
fig = plt.figure(figsize=(8, 6))
for timer in tqdm(range(0, 360, step), desc='Анимация расчёта по четырёхточечной схеме'):
    plt.clf()
    ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, t_grid, y, cmap='plasma')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('T')
    ax_3d.set_zlabel('U')
    # ax_3d.set_zlim(-30, 30)
    ax_3d.view_init(elev=0 + timer / 15, azim=timer)
    ax_3d.dist = 8
    plt.title('Численное решение уравнения переноса\nЧетырёхточечная схема')
    plt.draw()
    plt.gcf().canvas.flush_events()
    if saving:
        plt.savefig(f'Четырёхточечная схема/{timer}.png')
# plt.ioff()
y = np.transpose(y)

# ------------- СЧИТАЕМ ПО ТРЁХТОЧЕЧНОЙ СХЕМЕ -------------

x = np.linspace(X_END, X_START, Nx)
t = np.linspace(T_START, T_END, Nt)

u_2 = np.zeros((Nx, Nt))   # Пустой массив для проверки
u_2[:, 0] = -sin(x * pi)



@njit
def p(v):
    return -arctan(v) * v + log(sqrt(1 + v * v))


@njit
def f_2(v, c1, c2):
    return (v - c1) / tau + (p(c2) - p(v)) / h


@njit
def df(v):
    return 1 / tau - -arctan(v) / h


@njit
def solve(c1, c2):
    u1 = c2
    delta = eps + 1
    while delta > eps:
        u = u1
        u1 = u - f_2(u, c1, c2) / df(u)
        delta = abs(u1 - u)
    return u1


timer = time.time()

for i in tqdm(range(Nx - 2, -1, -1), desc='Расчёт по трёхточечной схеме'):
    for j in range(1, Nt):
        u_2[i, j] = solve(u_2[i, j - 1], u_2[i + 1, j])

u_2 = np.flip(u_2, axis=0)
print(f'Трёхточечная схема посчтина за {time.time()-timer} секунд')


x_grid, t_grid = np.meshgrid(xn, tm)
plt.ion()
fig = plt.figure(figsize=(8, 6))
if saving:
    step = 1
else:
    step = 10
for timer in tqdm(range(0, 360, step), desc='Анимация расчёта по трёхточечной схеме'):
    plt.clf()
    ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, t_grid, u_2.T, cmap='plasma')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('T')
    ax_3d.set_zlabel('U')
    ax_3d.view_init(elev=0 + timer / 15, azim=timer)
    ax_3d.dist = 8
    plt.title('Численное решение уравнения переноса\nТрёхточечная схема')
    plt.draw()
    plt.gcf().canvas.flush_events()
    if saving:
        plt.savefig(f'Трёхточечная схема/{timer}.png')


x_grid, t_grid = np.meshgrid(tm, xn)
frames = []
plt.ion()
fig = plt.figure(figsize=(8, 6))
if saving:
    step = 1
else:
    step = 10
for timer in tqdm(range(0, 360, step), desc='Анимация относительной ошибки'):
    plt.clf()
    ax_3d = plt.subplot(projection='3d')
    ax_3d.plot_surface(x_grid, t_grid, (y-u_2)/10, cmap='plasma')
    ax_3d.set_xlabel('T')
    ax_3d.set_ylabel('X')
    ax_3d.set_zlabel('U')
    # ax_3d.set_zlim(-30, 30)
    ax_3d.view_init(elev=0 + timer / 15, azim=timer)
    ax_3d.dist = 8
    plt.title('Численное решение уравнения переноса\nОтносительная ошибка')
    plt.draw()
    plt.gcf().canvas.flush_events()
    if saving:
        plt.savefig(f'Относительная ошибка/{timer}.png')

print(f'\nМаксимальная ошибка составила {np.max(abs(y - u_2))}')
print(f'\nМинимальная ошибка составила {np.min(abs(y - u_2))}')
print(f'\nСредняя по модулю ошибка составила {np.mean(abs(y - u_2))}')





# ------------- СОЗДАЁМ АНИМАЦИИ -------------
if saving:
    frames = []
    for t in range(360):
        image = imageio.v2.imread(f'Четырёхточечная схема/{t}.png')
        frames.append(image)
    imageio.mimsave('Четырёхточечная схема.gif',  # output gif
                    frames,  # array of input frames
                    fps=30)

    frames = []
    for t in range(360):
        image = imageio.v2.imread(f'Трёхточечная схема/{t}.png')
        frames.append(image)
    imageio.mimsave('Трёхточечная схема.gif',  # output gif
                    frames,  # array of input frames
                    fps=30)

    frames = []
    for t in range(360):
        image = imageio.v2.imread(f'Относительная ошибка/{t}.png')
        frames.append(image)
    imageio.mimsave('Относительная ошибка.gif',  # output gif
                    frames,  # array of input frames
                    fps=30)