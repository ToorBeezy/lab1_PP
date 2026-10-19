"""Гистограммы средних по строкам и столбцам (задача 4)

Делаем через matplotlib.pyplot.hist"""

import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_histograms(stats) -> None:
    """Гистограммы построчных и постолбцовых средних"""
    plt.figure(figsize=(10, 4)) # создаем холст
    plt.subplot(1, 2, 1)        # делаем сетку под два графика (1 строка, 2 столбца) и активируем первуый столбец для записи
    # рисуем первый график и обзываем оси
    plt.hist(stats.row_means, bins=20, edgecolor="black")
    plt.title("Средние по строкам")
    plt.xlabel("значение среднего")
    plt.ylabel("число строк")
    plt.subplot(1, 2, 2)        # активируем второй столбец\ячейку для записи
    # рисуем второй график и обзываем оси
    plt.hist(stats.column_means, bins=20, color="orange", edgecolor="black")
    plt.title("Средние по столбцам")
    plt.xlabel("значение среднего")
    plt.ylabel("число столбцов")
    plt.tight_layout()      # автоматически подгоняет элементы, чтобы не было ляпов
    plt.savefig("row_column_histograms.png", dpi=150)   # сохраняем как файл "row_column_histograms.png" с разрешением 1500х600
    plt.close()


if __name__ == "__main__":
    demo = RandomMatrixInput(rows=200, columns=50, mean=10.0, std=2.0)
    plot_histograms(matrix_statistics(demo))
    print("Saved: row_column_histograms.png")
