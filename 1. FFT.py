import pyfftw
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Считывание данных из файла и преобразование к np.array
with open('17.txt', mode='r') as f:
    data = f.readlines()[1:]

data = np.array([float(x) for x in data])
marks = np.linspace(int(-data.size / 2 + 1), int(data.size / 2), data.size, dtype='int8')


# Отрисовка графика
def show_graph(x: np.array, y: np.array):
    plt.plot(x, y)
    plt.grid()
    plt.show()


def fft(a: np.array):
    with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
        pyfftw.interfaces.cache.enable()
        return scipy.fft.fft(a, norm='forward')


def ifft(a: np.array):
    with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
        pyfftw.interfaces.cache.enable()
        return scipy.fft.ifft(a, norm='forward')


def print_data(data: np.array):
    for index, value in enumerate(data):
        # print(f'{marks[index]}  {value:.3f}')
        symbol = '+' if value.imag > 0 else '-'
        print(f'{value.real:.5f} {symbol} {np.abs(value.imag):.5f}j')


# [print(x) for x in data]

show_graph(np.arange(data.size), data)

data = fft(data)

print_data(np.roll(np.fft.fftshift(data), -1))

show_graph(marks, np.abs(np.fft.fftshift(data)) ** 2)
show_graph(marks, np.abs(np.roll(np.fft.fftshift(data), -1)) ** 2)

print(np.sum(data))

data = ifft(data)
print_data(data)

show_graph(marks, data)
