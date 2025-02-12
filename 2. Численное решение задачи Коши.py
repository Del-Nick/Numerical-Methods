import numpy as np
import matplotlib.pyplot as plt

u_0 = (50 + 150 + 32) / 30
v_0 = 0

# Схема устойчива при dt <= 1
dt = .5

t = np.arange(0, 20, dt)
u = t * 0
v = t * 0

t_analit = np.linspace(0, 20, 1000)
u_analit = u_0 * np.cos(t_analit)
v_analit = -u_0 * np.sin(t_analit)

EULER_SCHEME = True


def euler(num_steps: int) -> np.ndarray:
    '''
    Функция для поиска второго элемента массива значений
    :param num_steps: число, на которое нужно разделить шаг основной сетки
    :return: Возращает значпения вторых координаты и скорости
    '''

    dt_ = dt / num_steps
    v_, u_ = v_0, u_0

    # Ищем второй элемент
    for _ in np.arange(num_steps):
        u_ += v_ * dt_
        v_ -= u_ * dt_

    return u_, v_


def solving(num_steps: int = 1) -> np.array:
    '''
    Функция для решения схемой с перешагиванием
    :param num_steps: Число частей, на которое нужно разделить основной шаг сетки
    :return: :return: кортеж из двух массивов значений скорости и координат
    '''

    dt_ = dt if EULER_SCHEME else dt/num_steps
    t_ = np.arange(0, 20, dt_)
    u_ = t_ * 0
    v_ = t_ * 0
    u_[0], v_[0] = u_0, v_0

    print(f'dt = {dt_}')

    if EULER_SCHEME:
        u_[1], v_[1] = euler(num_steps)
    else:
        u_[1] = u_0 * np.cos(t_[1])
        v_[1] = -u_0 * np.sin(t_[1])

    for i in np.arange(t_.size - 2):
        u_[i + 2] = u_[i] + 2 * dt_ * v_[i + 1]
        v_[i + 2] = v_[i] - 2 * dt_ * u_[i + 1]
    return u_


if EULER_SCHEME:
    for num_steps in [1, 2]:
        print(f'num steps = {num_steps}')
        u = solving(num_steps)

        plt.figure(num=f'Решение по схеме Эйлера')
        plt.plot(t_analit, u_analit, label=f'Аналитическое решение c шагом dt = {dt:.2f}')
        plt.plot(t, u, label=f'Численное решение с шагом dt = {dt / num_steps:.2f}')

        plt.xlabel('t, с')
        plt.ylabel('u')
        plt.grid()
        plt.show()

        plt.figure(num=f'Ошибка в схеме Эйлера')
        step = int(u_analit.size / u.size)
        plt.plot(t, u_analit[::step] - u)
        plt.xlabel('t, с')
        plt.ylabel('u - u(аналитическое)')
        plt.grid()
        plt.show()

else:
    for num_steps in [1, 2, 5]:
        u = solving(num_steps)

        plt.figure(num=f'Точное решение')
        plt.plot(t_analit, u_analit,
                 label='Аналитическое решение')

        plt.plot(np.arange(0, 20, dt / num_steps), u,
                 label=f'Численное решение с шагом dt = {dt / num_steps:.2f}')

        plt.xlabel('t, с')
        plt.ylabel('u')
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure(num=f'График ошибки при dt = {dt / num_steps:.2f}')
        error = u_0 * np.cos(np.arange(0, 20, dt / num_steps)) - u
        plt.plot(np.arange(0, 20, dt / num_steps),
                 error)

        plt.xlabel('t, с')
        plt.ylabel('u - u(аналитическое)')
        plt.grid()
        plt.show()
