"""Заготовки задач на NumPy"""

"""Здесь только код решений, тесты находятся в test_numpy_tasks.py"""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)

### Задача 1. Сумма произведений матриц на векторы

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    """Сумма A1 @ x1 + A2 @ x2 + ... + Ap @ xp, результат - столбец (n, 1)"""
    matrices, vectors = data.matrices, data.vectors
    matrices = [np.asarray(matrix) for matrix in matrices]
    vectors = [np.asarray(vector).reshape(-1, 1) for vector in vectors]  # из вектора в столбец
    n = matrices[0].shape[0]
    result = np.zeros((n, 1)) # нулевой вектор(столбец)
    for matrix, vector in zip(matrices, vectors):
        result += matrix @ vector
    return result

### Задача 2. Бинаризация матрицы

def binarize(data: BinarizeInput) -> np.ndarray:
    """1 там, где элемент строго больше threshold, иначе 0"""
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    # сравнение даёт матрицу из True/False, astype превращает её в 1/0
    return (matrix > threshold).astype(int)

### Задача 3. Уникальные элементы строк и столбцов

def unique_rows(data: MatrixInput) -> list[list[float]]:
    """Список уникальных значений каждой строки"""
    matrix = np.asarray(data.matrix)
    # перебор по двумерному массиву идёт по строкам;
    # dict.fromkeys убирает повторы и сохраняет порядок первого вхождения
    return [list(dict.fromkeys(row.tolist())) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    """Список уникальных значений каждого столбца"""
    matrix = np.asarray(data.matrix)
    # столбец исходной матрицы - это строка транспонированной, поэтому сохраняем всю логику лишь транспонируя матрицу
    return [list(dict.fromkeys(column.tolist())) for column in matrix.T]

### Задача 4. Статистики случайной матрицы

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    """Случайная нормальная матрица (m, n) и её статистики по строкам и столбцам"""
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)  # seed - чтобы матрица была той же при перезапуске
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),        # по каждой строке
        column_means=matrix.mean(axis=0),     # по каждому столбцу
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )

### Задача 5. Шахматная матрица

def chess(data: ChessInput) -> np.ndarray:
    """Матрица (m, n) в шахматном порядке, левый верхний элемент равен a"""
    rows, columns, a, b = data.rows, data.columns, data.first, data.second
    row_indices = np.arange(rows).reshape(-1, 1)        # номера строк столбцом
    column_indices = np.arange(columns).reshape(1, -1)  # номера столбцов строкой
    # чётность суммы номеров: 0 - клетка a, 1 - клетка b
    checker = (row_indices + column_indices) % 2
    return np.where(checker == 0, a, b)

### Задача 6. Прямоугольник и эллипс

# прямоугольник

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    """RGB-изображение (m, n, 3) с прямоугольником width x height в центре"""
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color)
    # одинаковые поля сверху/снизу и слева/справа - гарантирует, что прямоугольник будет в центре
    top = (image_height - height) // 2 # ширина - столбцы, высота - строки
    left = (image_width - width) // 2
    image[top:top + height, left:left + width] = shape_color
    return image

# эллипс

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    """RGB-изображение (m, n, 3) с закрашенным эллипсом с полуосями a и b в центре"""
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    # x - номер столбца, y - номер строки
    x = np.arange(image_width).reshape(1, -1)
    y = np.arange(image_height).reshape(-1, 1)
    # считаем координаты центра эллипса
    x0 = (image_width - 1) / 2  # центр сетки пикселей 0..n-1
    y0 = (image_height - 1) / 2
    # левая часть уравнения эллипса из условия; <= 1 значит закрашен внутри и на границе
    inside = (x - x0) ** 2 / semi_axis_x ** 2 + (y - y0) ** 2 / semi_axis_y ** 2 <= 1
    image = np.full((image_height, image_width, 3), background_color)
    image[inside] = shape_color # fancy indexing
    return image

### Задача 7. Анализ временного ряда

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    """Статистики ряда: среднее, дисперсия, отклонение, экстремумы, скользящее среднее"""
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    mean, variance, std = values.mean(), values.var(), values.std()

    # кандидаты в экстремумы - все элементы, кроме первого и последнего (у них нет обоих соседей), и их соседи слева и справа
    # экстремумы ищем срезами
    middle = values[1:-1]
    left_neighbors = values[:-2]
    right_neighbors = values[2:]
    local_maxima = np.where((middle > left_neighbors) & (middle > right_neighbors))[0] + 1 # строгое сравнение по условию
    local_minima = np.where((middle < left_neighbors) & (middle < right_neighbors))[0] + 1 # прибавляем единицу. т.к срез начинается с элемента номер один
    # средние по всем полным окнам длины window
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid") # режим valid оставляет только полные окна, поэтому длина результата ровно n - p + 1, как и в условии
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )

### Задача 8. One-hot encoding

def one_hot(data: OneHotInput) -> np.ndarray:
    """One-hot представление меток: в строке i единица стоит на месте labels[i]"""
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:  # метки нумеруются с нуля
        class_count = labels.max() + 1
    result = np.zeros((labels.size, class_count), dtype=int)
    # в каждой строке i ставим единицу в столбец labels[i]
    result[np.arange(labels.size), labels] = 1 # ура опять fancy indexing
    return result
