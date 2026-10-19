"""Тесты к задачам из numpy_tasks.py"""

import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput
)
from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot  
)

def column(values):
    """Вектор-столбец (n, 1)."""
    return np.array(values, dtype=float).reshape(-1, 1)


### Задача 1. Сумма произведений матриц на векторы


def test_sum_prod_single_pair():
    result = sum_prod(MatrixVectorBatchInput(
        matrices=[np.eye(2, dtype=int)],
        vectors=[column([5, -3])],
    ))
    assert result.shape == (2, 1)  # проверка, что результат - столбец. Напомню, что в нашем решении мы преобразуем вектор в столбец
    assert np.array_equal(result, column([5, -3]))  # единичная матрица по факту ничего не меняет


def test_sum_prod_two_pairs():
    a1 = np.array([[1, 2], [3, 4]])
    x1 = column([1, 0])  # A1 @ x1 = [1, 3]
    a2 = np.array([[0, 1], [1, 0]])
    x2 = column([0, 1])  # A2 @ x2 = [1, 0]
    result = sum_prod(MatrixVectorBatchInput(matrices=[a1, a2], vectors=[x1, x2]))
    assert np.array_equal(result, column([2, 3]))


def test_sum_prod_flat_vectors():
    # плоский вектор (n,) тоже принимается и приводится к столбцу
    result = sum_prod(MatrixVectorBatchInput(
        matrices=[np.array([[1, 2], [3, 4]])],
        vectors=[[1, 1]],
    ))
    assert result.shape == (2, 1)
    assert np.array_equal(result, column([3, 7])) # 1*1 + 2*1 = 3; 3*1 + 4*1 = 7 -> [3, 7]


def test_sum_prod_three_pairs():
    a1 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])  # диагональная матрица
    x1 = column([1, 0, -1])  # A1 @ x1 = [1, 0, -3]
    a2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # по факту меняет первые две координаты местами
    x2 = column([2, 3, 4])  # A2 @ x2 = [3, 2, 4]
    a3 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])  # ненулевым будет только первый элемент
    x3 = column([1, 2, 3])  # A3 @ x3 = [6, 0, 0]
    result = sum_prod(MatrixVectorBatchInput(matrices=[a1, a2, a3], vectors=[x1, x2, x3]))
    assert np.array_equal(result, column([10, 2, 1]))


### Задача 2. Бинаризация матрицы


def test_binarize_basic():
    matrix = np.array([[1, 2], [3, 4]])
    expected = np.array([[0, 0], [1, 1]])
    assert np.array_equal(binarize(BinarizeInput(matrix=matrix, threshold=2)), expected)


def test_binarize_equal_threshold():
    # элемент, равный порогу, единицу не даёт, т.к сравнение строгое
    matrix = np.array([[2.0, 2.5]])
    expected = np.array([[0, 1]])
    assert np.array_equal(binarize(BinarizeInput(matrix=matrix, threshold=2.0)), expected)


def test_binarize_negative_threshold():
    matrix = np.array([[-3, -1, 0]])
    expected = np.array([[0, 1, 1]])
    assert np.array_equal(binarize(BinarizeInput(matrix=matrix, threshold=-2)), expected)


def test_binarize_all_equal():
    # вся матрица стоит на пороге - везде нули
    matrix = np.array([[5, 5], [5, 5]])
    assert np.array_equal(
        binarize(BinarizeInput(matrix=matrix, threshold=5)), np.zeros((2, 2), dtype=int)
    )


def test_binarize_default_threshold():
    # threshold не передан - используется значение по умолчанию указанное в grader contracts (float: 0.5)
    matrix = np.array([[0.2, 0.7, 0.5]])
    result = binarize(BinarizeInput(matrix=matrix))
    assert np.array_equal(result, np.array([[0, 1, 0]]))

def test_binarize_input_not_modified():
    # функция возвращает новую матрицу, вход не меняет как и задано по условию
    matrix = np.array([[1, 5], [7, 2]])
    binarize(BinarizeInput(matrix=matrix, threshold=3))
    assert np.array_equal(matrix, np.array([[1, 5], [7, 2]]))


### Задача 3. Уникальные элементы строк и столбцов


def test_unique_basic():
    # порядок первого вхождения: [2, 2, 1] даёт [2, 1], а не отсортированный [1, 2]
    matrix = np.array([[1, 1, 2], [2, 2, 1], [3, 0, 3]])
    assert unique_rows(MatrixInput(matrix=matrix)) == [[1, 2], [2, 1], [3, 0]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[1, 2, 3], [1, 2, 0], [2, 1, 3]]


def test_unique_negatives():
    matrix = np.array([[-1, -1, 4], [4, -1, 4]])  # отрицательные и повторы
    assert unique_rows(MatrixInput(matrix=matrix)) == [[-1, 4], [4, -1]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[-1, 4], [-1], [4]]


def test_unique_floats():
    matrix = np.array([[1.5, 1.5, 2.5], [0.5, 0.5, 0.5]])
    assert unique_rows(MatrixInput(matrix=matrix)) == [[1.5, 2.5], [0.5]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[1.5, 0.5], [1.5, 0.5], [2.5, 0.5]]


def test_unique_single_column():
    matrix = np.array([[3], [1], [3]])  # матрица с одним столбцом
    assert unique_rows(MatrixInput(matrix=matrix)) == [[3], [1], [3]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[3, 1]]


def test_unique_all_same():
    # все значения одинаковые - в каждой строке и столбце по одному числу
    matrix = np.array([[7, 7], [7, 7]])
    assert unique_rows(MatrixInput(matrix=matrix)) == [[7], [7]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[7], [7]]


def test_unique_single_row():
    # матрица из одной строки: каждый столбец - список из одного элемента
    matrix = np.array([[2, 1, 2]])
    assert unique_rows(MatrixInput(matrix=matrix)) == [[2, 1]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[2], [1], [2]]


### Задача 4. Статистики случайной матрицы


def test_matrix_statistics_seed():
    # тестируем рнг и сидированность
    first = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=0.0, std=1.0, seed=7))
    second = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=0.0, std=1.0, seed=7))
    other = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=0.0, std=1.0, seed=8))
    assert np.array_equal(first.matrix, second.matrix)  # один seed - одна матрица
    assert not np.array_equal(first.matrix, other.matrix) # разный seed - разные матрицы


def test_matrix_statistics_shapes():
    stats = matrix_statistics(RandomMatrixInput(rows=6, columns=5, mean=2.0, std=3.0, seed=0))
    assert stats.matrix.shape == (6, 5)
    assert stats.row_means.shape == (6,)
    assert stats.column_means.shape == (5,)
    assert stats.row_variances.shape == (6,)
    assert stats.column_variances.shape == (5,)
    # assert_allclose здесь, а не array_equal, потому что
    # числа вещественные и результат бы в любом случае не мог бы быть идентичный.
    # Смысл ведь в том, что какие бы числа ни выпали, статистика ДОЛЖНА
    # быть средней/дисперсией именно этой матрицы, а не какой-то другой
    np.testing.assert_allclose(stats.row_means, stats.matrix.mean(axis=1))
    np.testing.assert_allclose(stats.column_means, stats.matrix.mean(axis=0))
    np.testing.assert_allclose(stats.row_variances, stats.matrix.var(axis=1))
    np.testing.assert_allclose(stats.column_variances, stats.matrix.var(axis=0))


def test_matrix_statistics_distribution():
    # на большой матрице выборочные среднее и отклонение близки к заданным
    stats = matrix_statistics(RandomMatrixInput(rows=200, columns=50, mean=5.0, std=2.0, seed=0))
    assert stats.matrix.shape == (200, 50)
    assert stats.matrix.mean() == pytest.approx(5.0, abs=0.1) # задал точность до 0.1, как выше написал отклонения "близки", но разумеется не идеальны
    assert stats.matrix.std() == pytest.approx(2.0, abs=0.1)


### Задача 5. Шахматная матрица


def test_chess_readme_example():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    # print(result)
    assert np.array_equal(result, expected)


def test_chess_one_cell():
    result = chess(ChessInput(rows=1, columns=1, first=5, second=9))
    # print(result)
    assert np.array_equal(result, [[5]])


def test_chess_one_row():
    result = chess(ChessInput(rows=1, columns=5, first=0, second=1))
    # print(result)
    assert np.array_equal(result, [[0, 1, 0, 1, 0]])


def test_chess_one_column():
    result = chess(ChessInput(rows=5, columns=1, first=0, second=1))
    # print(result)
    assert np.array_equal(result, [[0], [1], [0], [1], [0]])


def test_chess_first_is_top_left(): # в задаче указан элемент a, в классе описан как "first", тут буду так и обзывать
    result = chess(ChessInput(rows=2, columns=2, first=7, second=-7))
    # print(result)
    assert np.array_equal(result, [[7, -7], [-7, 7]])


def test_chess_adjacent_cells_differ():
    # на произвольном размере: соседи по вертикали и горизонтали различаются
    result = chess(ChessInput(rows=4, columns=7, first=3, second=8))
    # print(result)
    assert result[0, 0] == 3  # левый верхний - 3
    for i in range(4):
        for j in range(7):
            assert result[i, j] in (3, 8)  # только два заданных значения
            if j + 1 < 7:
                assert result[i, j] != result[i, j + 1]  # сосед справа
            if i + 1 < 4:
                assert result[i, j] != result[i + 1, j]  # сосед снизу


### Задача 6. Прямоугольник и эллипс

SHAPE = (255, 0, 0)       # красный - цвет фигуры
BACKGROUND = (0, 0, 255)  # синий - цвет фона

# прямоугольник

def test_draw_rectangle_centered():
    result = draw_rectangle(RectangleInput(
        width=3, height=3, image_height=5, image_width=7,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    assert result.shape == (5, 7, 3)
    # прямоугольник в центре: строки 1..3, столбцы 2..4
    for i in range(5):
        for j in range(7):
            expected = SHAPE if (1 <= i <= 3 and 2 <= j <= 4) else BACKGROUND
            assert tuple(result[i, j]) == expected


def test_draw_rectangle_orientation():
    # width идёт по столбцам, height - по строкам
    result = draw_rectangle(RectangleInput(
        width=4, height=1, image_height=5, image_width=7,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    for i in range(5):
        for j in range(7):
            expected = SHAPE if (i == 2 and 1 <= j <= 4) else BACKGROUND
            assert tuple(result[i, j]) == expected


def test_draw_rectangle_full_image():
    # прямоугольник размером с всё изображение, закрашенный целиком
    result = draw_rectangle(RectangleInput(
        width=4, height=3, image_height=3, image_width=4,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    for i in range(3):
        for j in range(4):
            assert tuple(result[i, j]) == SHAPE


def test_draw_rectangle_one_pixel():
    # прямоугольник 1x1 попадает ровно в центр изображения
    result = draw_rectangle(RectangleInput(
        width=1, height=1, image_height=5, image_width=7,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    for i in range(5):
        for j in range(7):
            expected = SHAPE if (i == 2 and j == 3) else BACKGROUND
            assert tuple(result[i, j]) == expected


# эллипс


def test_draw_ellipse_unit_circle():
    result = draw_ellipse(EllipseInput(
        semi_axis_x=1, semi_axis_y=1, image_height=5, image_width=5,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    # круг радиуса 1 с центром (2, 2): сам центр и четыре соседа
    shape_mask = np.all(result == np.array(SHAPE), axis=2)
    assert shape_mask.sum() == 5
    for i, j in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]:
        assert shape_mask[i, j]


def test_draw_ellipse_even_image():
    result = draw_ellipse(EllipseInput(
        semi_axis_x=1, semi_axis_y=1, image_height=4, image_width=4,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    # при четной стороне центр (1.5, 1.5) - закрашены четыре центральных пикселя
    shape_mask = np.all(result == np.array(SHAPE), axis=2)
    assert shape_mask.sum() == 4


def test_draw_ellipse_stretched():
    result = draw_ellipse(EllipseInput(
        semi_axis_x=3, semi_axis_y=1, image_height=5, image_width=9,
        shape_color=SHAPE, background_color=BACKGROUND,
    ))
    shape_mask = np.all(result == np.array(SHAPE), axis=2)
    assert shape_mask.sum() == 9  # 7 пикселей в средней строке + по одному сверху и снизу
    for j in range(1, 8):  # средняя строка: от центра минус 3 до центр плюс 3
        assert shape_mask[2, j]
    for i in [1, 3]:  # соседние строки: единственный пиксель в центральном столбце
        assert shape_mask[i, 4]
        assert shape_mask[i].sum() == 1


### Задача 7. Анализ временного ряда


def test_time_series_basic():
    stats = analyze_time_series(TimeSeriesInput(values=[1, 3, 2, 5, 4], window=2))
    assert stats.mean == pytest.approx(3.0)
    assert stats.variance == pytest.approx(2.0)
    assert stats.std == pytest.approx(np.sqrt(2.0))
    np.testing.assert_array_equal(stats.local_maxima_indices, [1, 3])  # 3 и 5
    np.testing.assert_array_equal(stats.local_minima_indices, [2])
    np.testing.assert_allclose(stats.moving_average, [2.0, 2.5, 3.5, 4.5])
    assert stats.moving_average.size == 4  # n - window + 1


def test_time_series_plateau():
    # плато не даёт экстремума: 2 не строго больше 2
    stats = analyze_time_series(TimeSeriesInput(values=[1, 2, 2, 3, 1], window=3))
    np.testing.assert_array_equal(stats.local_maxima_indices, [3])
    np.testing.assert_array_equal(stats.local_minima_indices, [])
    np.testing.assert_allclose(stats.moving_average, [5 / 3, 7 / 3, 2.0])


def test_time_series_edges():
    # первый и последний элементы экстремумами не считаются, писал об этом в коде решения
    stats = analyze_time_series(TimeSeriesInput(values=[9, 0, 9], window=1))
    np.testing.assert_array_equal(stats.local_maxima_indices, [])
    np.testing.assert_array_equal(stats.local_minima_indices, [1])


def test_time_series_monotonic():
    # в монотонном ряде нет ни максимумов, ни минимумов
    stats = analyze_time_series(TimeSeriesInput(values=[1, 2, 3, 4, 5], window=2))
    np.testing.assert_array_equal(stats.local_maxima_indices, [])
    np.testing.assert_array_equal(stats.local_minima_indices, [])
    assert stats.mean == pytest.approx(3.0)
    assert stats.variance == pytest.approx(2.0)
    np.testing.assert_allclose(stats.moving_average, [1.5, 2.5, 3.5, 4.5])


def test_time_series_constant():
    # постоянный ряд: дисперсия и отклонение нулевые, экстремумов нет
    stats = analyze_time_series(TimeSeriesInput(values=[4, 4, 4, 4], window=2))
    assert stats.mean == pytest.approx(4.0)
    assert stats.variance == pytest.approx(0.0)
    assert stats.std == pytest.approx(0.0)
    np.testing.assert_array_equal(stats.local_maxima_indices, [])
    np.testing.assert_array_equal(stats.local_minima_indices, [])
    np.testing.assert_allclose(stats.moving_average, [4.0, 4.0, 4.0])


def test_time_series_alternating():
    # зигзаг даёт максимум и минимум в каждой «волне»
    stats = analyze_time_series(TimeSeriesInput(values=[1, 3, 1, 3, 1], window=1))
    np.testing.assert_array_equal(stats.local_maxima_indices, [1, 3])
    np.testing.assert_array_equal(stats.local_minima_indices, [2])
    np.testing.assert_allclose(stats.moving_average, [1, 3, 1, 3, 1])  # окно 1 - ряд не меняется


def test_time_series_window_equals_length():
    # окно длиной в весь ряд - скользящее среднее состоит из одного числа
    stats = analyze_time_series(TimeSeriesInput(values=[2, 4, 6, 8], window=4))
    assert stats.moving_average.size == 1
    assert stats.moving_average[0] == pytest.approx(5.0)


### Задача 8. One-hot encoding


def test_one_hot_basic():
    result = one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=None))
    assert np.array_equal(result, np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]))
    assert result.shape == (4, 4)  # столбцов max(labels) + 1


def test_one_hot_explicit_class_count():
    result = one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=6))
    assert result.shape == (4, 6)  # лишние столбцы нулевые
    assert np.array_equal(result, np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]))


def test_one_hot_single_label():
    result = one_hot(OneHotInput(labels=[1], class_count=None))
    assert np.array_equal(result, np.array([[0, 1]]))


def test_one_hot_all_same_labels():
    # одна и та же метка несколько раз = одинаковые строки
    result = one_hot(OneHotInput(labels=[2, 2, 2], class_count=None))
    assert result.shape == (3, 3)
    assert np.array_equal(result, np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]))


def test_one_hot_missing_classes():
    # метки 1-3 не встретились, но столбцы для них должны остаться
    result = one_hot(OneHotInput(labels=[4, 0], class_count=None))
    assert result.shape == (2, 5)  # max(labels) + 1 = 5
    assert np.array_equal(result, np.array([
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ]))


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items())
             if name.startswith("test_") and callable(f)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASSED")
        except Exception as error:
            print(f"{test.__name__}: FAILED")
            print(f"    {type(error).__name__}: {error}")
            failed += 1
    print(f"\n{len(tests) - failed} из {len(tests)} тестов пройдено")
