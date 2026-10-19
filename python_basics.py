"""Заготовки задач на базовый Python"""

"""Здесь только код решений, тесты находятся в test_python_basics.py"""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

### Задача 1. Подсчёт гласных

def count_vowels(data: TextInput) -> int:
    """Количество гласных a, e, i, o, u в строке (регистр не важен)"""
    text = data.value
    count = 0
    for letter in text.lower():
        if letter in "aeiou":
            count += 1
    return count

### Задача 2. Уникальные символы

def has_unique_characters(data: TextInput) -> bool:
    """True, если каждый символ строки встречается не более одного раза"""
    text = data.value
    seen = set()
    for letter in text:
        if letter in seen:  # такой символ уже был - дальше можно не идти
            return False
        seen.add(letter)
    return True

### Задача 3. Единичные биты

def count_one_bits(data: PositiveIntegerInput) -> int:
    """Количество единиц в двоичной записи числа"""
    number = data.value
    count = 0
    while number > 0:
        count += number % 2  # младший бит текущего числа
        number //= 2  # отбрасываем его и переходим к следующему
    return count

### Задача 4. Мультипликативная устойчивость

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    """Сколько раз нужно перемножить цифры числа, чтобы осталось однозначное"""
    number = data.value
    steps = 0
    while number > 9:
        multi = 1
        for digit in str(number): # берем по цифре из строчного представления изначального числа
            multi *= int(digit)   # перемножаем их
        number = multi            # и записываем как новое число
        steps += 1                # плюс считаем количество таких иттераций, что и является ответом на задачу
    return steps

### Задача 5. Средняя квадратичная ошибка

def mse(data: VectorPairInput) -> float:
    """Средняя квадратичная ошибка между двумя векторами"""
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы должны иметь одинаковую длину")
    squared_sum = 0
    for x, y in zip(predicted, expected):
        squared_sum += (x - y) ** 2
    return squared_sum / len(predicted)

### Задача 6. Разложение на простые множители

def prime_factorization(data: PositiveIntegerInput) -> str:
    """Строка с разложением на простые множители, например (2**5)(5)(7**2)(11)"""
    remaining = data.value
    factors = {}  # словарь: простой множитель -> его степень
    divisor = 2
    # делитель достаточно проверять до корня текущего остатка
    while divisor * divisor <= remaining: # граница по текущему остатку
        while remaining % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            remaining //= divisor
        divisor += 1
    if remaining > 1:  # то, что осталось - само по себе является простым множителем
        factors[remaining] = factors.get(remaining, 0) + 1
    result = ""
    for prime in sorted(factors):  # множители по возрастанию по условию
        if factors[prime] == 1:
            result += f"({prime})" # первую степень не выводим по условию
        else:
            result += f"({prime}**{factors[prime]})"
    return result

### Задача 7. Пирамида из кубиков

def pyramid(data: PositiveIntegerInput) -> int | str:
    """Высота пирамиды, если n = 1^2 + ... + k^2, иначе строку It is impossible"""
    cube_count = data.value # n - по условию, поменял для читаемости
    total = 0
    level = 0
    while total < cube_count:  # достраиваем уровни, пока кубиков меньше n
        level += 1
        total += level * level
    if total == cube_count:
        return level
    return "It is impossible"

### Задача 8. Сбалансированное число

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    """True, если суммы цифр слева и справа от середины числа равны

    При нечётной длине средняя цифра не входит в суммы,
    при чётной - две средние цифры не входят в суммы"""

    number = data.value
    digits = str(number)
    half = len(digits) // 2
    if len(digits) % 2 == 0:  # выбрасываем две средние цифры
        left = digits[: half - 1]
        right = digits[half + 1 :]
    else:  # выбрасываем одну среднюю
        left = digits[:half]
        right = digits[half + 1 :] # Если учитывать, что например 5 // 2 = 2, то отбрасывать надо именно от левой части, т.к там уже будет 2 числа, а в правой части, без отбрасывания, 3 числа, поэтому отступаем от левой части
    left_sum = 0
    for digit in left:
        left_sum += int(digit)
    right_sum = 0
    for digit in right:
        right_sum += int(digit)
    return left_sum == right_sum
