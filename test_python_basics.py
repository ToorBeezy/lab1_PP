"""Тесты к задачам из python_basics.py"""

import pytest

from grader_contracts.python_basics import (
    TextInput,
    PositiveIntegerInput,
    VectorPairInput
)
from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number
)


### Задача 1. Подсчёт гласных


def test_count_vowels():
    assert count_vowels(TextInput("hello")) == 2
    assert count_vowels(TextInput("programming")) == 3
    assert count_vowels(TextInput("AEIOU")) == 5  # регистр не важен
    assert count_vowels(TextInput("AbRaCaDaBrA")) == 5
    assert count_vowels(TextInput("")) == 0
    assert count_vowels(TextInput("bcdfg")) == 0
    assert count_vowels(TextInput("my")) == 0  # y не входит в гласные из условия
    assert count_vowels(TextInput("aaaa")) == 4
    assert count_vowels(TextInput("queue")) == 4


### Задача 2. Уникальные символы


def test_has_unique_characters():
    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("aab")) is False
    assert has_unique_characters(TextInput("aba")) is False
    assert has_unique_characters(TextInput("")) is True
    assert has_unique_characters(TextInput("a")) is True
    assert has_unique_characters(TextInput("aa")) is False
    assert has_unique_characters(TextInput("Aa")) is True  # A и a - разные символы
    assert has_unique_characters(TextInput("AaA")) is False
    assert has_unique_characters(TextInput("a b c")) is False  # два пробела
    assert has_unique_characters(TextInput("a bc")) is True
    assert has_unique_characters(TextInput("12345")) is True  # цифры тоже символы
    assert has_unique_characters(TextInput("abcdefga")) is False


### Задача 3. Единичные биты


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(2)) == 1  # 10
    assert count_one_bits(PositiveIntegerInput(3)) == 2  # 11
    assert count_one_bits(PositiveIntegerInput(10)) == 2  # 1010
    assert count_one_bits(PositiveIntegerInput(13)) == 3  # 1101
    assert count_one_bits(PositiveIntegerInput(255)) == 8  # 11111111
    assert count_one_bits(PositiveIntegerInput(1023)) == 10 # 1111111111
    assert count_one_bits(PositiveIntegerInput(2**100)) == 1
    assert count_one_bits(PositiveIntegerInput(2**100 - 1)) == 100


def test_count_one_bits_matches_bin():
    # bin(n) уже дает двоичную запись - можно просто посчитать единицы в ней
    for n in [1, 7, 42, 128, 255, 1000000, 2**50 + 12345]:
        assert count_one_bits(PositiveIntegerInput(n)) == bin(n).count("1")


### Задача 4. Мультипликативная устойчивость


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3  # 27 -> 14 -> 4
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4  # 729->126->12->2
    assert multiplicative_persistence(PositiveIntegerInput(1)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(9)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1  # 1*0 = 0
    assert multiplicative_persistence(PositiveIntegerInput(11)) == 1  # 1*1 = 1
    assert multiplicative_persistence(PositiveIntegerInput(25)) == 2  # 10 -> 0
    assert multiplicative_persistence(PositiveIntegerInput(77)) == 4  # 49->36->18->8
    assert multiplicative_persistence(PositiveIntegerInput(679)) == 5 # 378->168->48->32->6
    
    # наличие нуля в числе - заканчивает перемножение в один ход, неважно сколь большое число
    assert multiplicative_persistence(PositiveIntegerInput(int(str(10**1000-1) + "0"))) == 1 # тут мы к тысяче девяток добавили всего один нолик и всё, мультипликативная стойкость остановилась на единице

    # минимальное известное число с максимальной мультипликативной стойкостью (на данный момент)
    assert multiplicative_persistence(PositiveIntegerInput(277777788888899)) == 11 # 4996238671872->438939648->4478976->338688->27648->2688->768->336->54->20->0


def test_multiplicative_persistence_bruteforce():
    # сверка с вариантом, где цифры разбираются делением, а не строкой
    # этот вариант более математический, а тот мне понравился больше из-за своей компактности :)
    # но в этом примере уменьшается сложность, т.к нет преобразования в строку
    for n in range(1, 100000):
        remaining, steps = n, 0
        while remaining > 9:
            product = 1
            while remaining > 0:
                product *= remaining % 10
                remaining //= 10
            remaining = product
            steps += 1
        assert multiplicative_persistence(PositiveIntegerInput(n)) == steps


### Задача 5. Средняя квадратичная ошибка


def test_mse():
    assert mse(VectorPairInput(predicted=[1, 2, 3], expected=[1, 2, 3])) == 0.0
    assert mse(VectorPairInput(predicted=[1], expected=[2])) == 1.0
    assert mse(VectorPairInput(predicted=[0, 0], expected=[3, 4])) == 12.5  # (9 + 16) / 2
    assert mse(VectorPairInput(predicted=[-1, 1], expected=[1, -1])) == 4.0
    assert mse(VectorPairInput(predicted=[1, 2], expected=[0, 4])) == 2.5
    result = mse(VectorPairInput(predicted=[10**9], expected=[-10**9]))
    assert result == float((10**9 - (-10**9)) ** 2) # 4*10**18
    assert isinstance(result, float)  # по условию результат - float


def test_mse_length_mismatch():
    with pytest.raises(ValueError):
        mse(VectorPairInput(predicted=[1, 2], expected=[1, 2, 3]))


def test_mse_empty():
    # пустые векторы дают 0/0 - питон бросает ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        mse(VectorPairInput(predicted=[], expected=[]))


### Задача 6. Разложение на простые множители


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(4)) == "(2**2)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"
    assert prime_factorization(PositiveIntegerInput(60)) == "(2**2)(3)(5)"
    assert prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)"
    assert prime_factorization(PositiveIntegerInput(720)) == "(2**4)(3**2)(5)"
    assert prime_factorization(PositiveIntegerInput(30030)) == "(2)(3)(5)(7)(11)(13)"
    assert prime_factorization(PositiveIntegerInput(2**10)) == "(2**10)"
    assert prime_factorization(PositiveIntegerInput(1000000007)) == "(1000000007)"
    assert prime_factorization(PositiveIntegerInput(1)) == ""  # у единицы множителей нет (единица не является простым числом)


### Задача 7. Пирамида из кубиков


def test_pyramid():
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2  # 1 + 4
    assert pyramid(PositiveIntegerInput(14)) == 3 # 1 + 4 + 9
    assert pyramid(PositiveIntegerInput(30)) == 4 # 1 + 4 + 9 + 16
    assert pyramid(PositiveIntegerInput(55)) == 5 #1 + 4 + 9 + 16 + 25
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(4)) == "It is impossible"  # 1 < 4, но 1+4 = 5
    assert pyramid(PositiveIntegerInput(13)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(15)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(338350)) == 100  # 1^2 + ... + 100^2
    assert pyramid(PositiveIntegerInput(338349)) == "It is impossible"  # на 1 меньше


def test_pyramid_formula():
    # по факту это задачка на представление числа в виде суммы квадратов
    # k(k+1)(2k+1)/6 - сумма квадратов по формуле
    for k in range(1, 51):
        total = k * (k + 1) * (2 * k + 1) // 6
        assert pyramid(PositiveIntegerInput(total)) == k


### Задача 8. Сбалансированное число


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True # 1+2+3 = 0+0+6
    assert is_balanced_number(PositiveIntegerInput(123456)) is False # 1+2 != 5+6
    assert is_balanced_number(PositiveIntegerInput(12321)) is True # 1+2 = 2+1
    assert is_balanced_number(PositiveIntegerInput(123321)) is True # 1+2 = 2+1
    assert is_balanced_number(PositiveIntegerInput(1712)) is False # 1 != 2
    assert is_balanced_number(PositiveIntegerInput(1000001)) is True # 1+0+0 = 0+0+1 
    assert is_balanced_number(PositiveIntegerInput(1234007)) is False # 1+2+3 != 0+0+7
    assert is_balanced_number(PositiveIntegerInput(1000)) is False # 1 != 0
    # однозначные и двузначные: середина съедает все цифры,
    # сравниваются пустые суммы, они равны
    assert is_balanced_number(PositiveIntegerInput(1)) is True
    assert is_balanced_number(PositiveIntegerInput(9)) is True
    assert is_balanced_number(PositiveIntegerInput(10)) is True
    assert is_balanced_number(PositiveIntegerInput(99)) is True
    assert is_balanced_number(PositiveIntegerInput(10**15 + 1)) is True # 1 000 000 000 000 001; 1=1


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items())
             if name.startswith("test_") and callable(f)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASSED")
        except Exception:
            print(f"{test.__name__}: FAILED")
            failed += 1
    print(f"\n{len(tests) - failed} из {len(tests)} python_basics тестов пройдено")