import torch

def create_tensors():
    """
    Создание различных тензоров для задания 1.1

    Returns:
        tuple: Кортеж из четырех тензоров:
            - random_tensor (3x4): заполнен случайными числами от 0 до 1
            - zeros_tensor (2x3x4): заполнен нулями
            - ones_tensor (5x5): заполнен единицами
            - range_tensor (4x4): заполнен числами от 0 до 15
    """
    # Тензор размером 3x4, заполненный случайными числами от 0 до 1
    random_tensor = torch.rand(3, 4)

    # Тензор размером 2x3x4, заполненный нулями
    zeros_tensor = torch.zeros(2, 3, 4)

    # Тензор размером 5x5, заполненный единицами
    ones_tensor = torch.ones(5, 5)

    # Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
    range_tensor = torch.arange(0, 16).reshape(4, 4)

    return random_tensor, zeros_tensor, ones_tensor, range_tensor


def tensor_operations(A, B):
    """
    Выполнение операций с тензорами для задания 1.2

    Args:
        A (torch.Tensor): Тензор A размером 3x4
        B (torch.Tensor): Тензор B размером 4x3

    Returns:
        tuple: Кортеж результатов операций:
            - A_transposed: транспонированный тензор A
            - matmul_result: результат матричного умножения A и B
            - element_result: поэлементное умножение A и B.T
            - sum_A: сумма всех элементов A
    """
    # Проверка размерностей тензоров
    if A.shape != (3, 4):
        raise ValueError("Тензор A должен иметь размерность 3x4")
    if B.shape != (4, 3):
        raise ValueError("Тензор B должен иметь размерность 4x3")

    # Транспонирование тензора A
    A_transposed = A.T

    # Матричное умножение A и B
    matmul_result = torch.matmul(A, B)

    # Поэлементное умножение A и транспонированного B
    element_result = A * B.T

    # Вычислите сумму всех элементов тензора A
    sum_A = torch.sum(A)

    return A_transposed, matmul_result, element_result, sum_A


def tensor_indexing():
    """
    Работа с индексацией и срезами тензоров для задания 1.3

    Returns:
        tuple: Кортеж результатов индексации:
            - first_row: первая строка тензора 5x5x5
            - last_column: последний столбец тензора
            - center_submatrix: подматрица 2x2 из центра
            - even_indices: элементы с четными индексами
    """
    # Создаем тензор 5x5x5 со случайными значениями
    tensor = torch.rand(5, 5, 5)

    # Извлеките первую строку
    first_row = tensor[0, :, :]

    # Извлеките последний столбец
    last_column = tensor[:, :, -1]

    # Извлеките подматрицу размером 2х2 из центра тензора
    # Предполагаем, что берем из каждого слоя по 2x2 из центра
    center_submatrix = tensor[:, 1:3, 1:3]  # Берем со всех 5 слоев

    # Извлеките все элементы с четными индексами
    even_indices = tensor[::2, ::2, ::2]

    return first_row, last_column, center_submatrix, even_indices


def tensor_reshaping():
    """
    Работа с формами тензора для задания 1.4

    Returns:
        dict: Словарь с тензорами разных форм:
            - shape_2x12: тензор 2x12
            - shape_3x8: тензор 3x8
            - shape_4x6: тензор 4x6
            - shape_2x3x4: тензор 2x3x4
            - shape_2x2x2x3: тензор 2x2x2x3
    """
    # Создаем тензор размером 24 элемента
    original_tensor = torch.arange(24)

    # Проверка, что тензор действительно содержит 24 элемента
    if original_tensor.numel() != 24:
        raise ValueError("Исходный тензор должен содержать 24 элемента")

    # Преобразуем его в формы 2x12
    shape_2x12 = original_tensor.reshape(2, 12)

    # Преобразуем его в формы 3x8
    shape_3x8 = original_tensor.reshape(3, 8)

    # Преобразуем его в формы 4x6
    shape_4x6 = original_tensor.reshape(4, 6)

    # Преобразуем его в формы 2x3x4
    shape_2x3x4 = original_tensor.reshape(2, 3, 4)

    # Преобразуем его в формы 2x2x2x3
    shape_2x2x2x3 = original_tensor.reshape(2, 2, 2, 3)

    return {
        'shape_2x12': shape_2x12,
        'shape_3x8': shape_3x8,
        'shape_4x6': shape_4x6,
        'shape_2x3x4': shape_2x3x4,
        'shape_2x2x2x3': shape_2x2x2x3
    }


def test_functions():
    """Тестирование всех функций"""
    print("=== Тестирование создания тензоров ===")
    random_tensor, zeros_tensor, ones_tensor, range_tensor = create_tensors()
    print("\n1.1.1 Случайный тензор 3x4:\n", random_tensor)
    print("\n1.1.2 Тензор нулей 2x3x4:\n", zeros_tensor)
    print("\n1.1.3 Тензор единиц 5x5:\n", ones_tensor)
    print("\n1.1.4 Тензор 4x4 с числами 0-15:\n", range_tensor)

    print("\n=== Тестирование операций с тензорами ===")
    A = torch.arange(12).reshape(3, 4).float()
    B = torch.arange(12).reshape(4, 3).float()
    print("\nТензор A:\n", A)
    print("\nТензор B:\n", B)

    A_transposed, matmul_result, element_result, sum_A = tensor_operations(A, B)
    print("\n1.2.1 Транспонированный A:\n", A_transposed)
    print("\n1.2.2 Матричное умножение A и B:\n", matmul_result)
    print("\n1.2.3 Поэлементное умножение A и B.T:\n", element_result)
    print("\n1.2.4 Сумма всех элементов A:", sum_A.item())

    print("\n=== Тестирование индексации тензоров ===")
    first_row, last_column, center_submatrix, even_indices = tensor_indexing()
    print("\n1.3.1 Первая строка тензора 5x5x5:\n", first_row)
    print("\n1.3.2 Последний столбец тензора 5x5x5:\n", last_column)
    print("\n1.3.3 Подматрица 2x2 из центра:\n", center_submatrix)
    print("\n1.3.4 Элементы с четными индексами:\n", even_indices)

    print("\n=== Тестирование изменения формы тензора ===")
    reshaped_tensors = tensor_reshaping()
    for shape, tensor in reshaped_tensors.items():
        print(f"\n1.4 Тензор формы {shape}:\n", tensor)
        print("Сумма элементов:", torch.sum(tensor).item())


if __name__ == "__main__":
    # Проверяем доступность GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Переносим все тензоры на выбранное устройство
    torch.set_default_device(device)

    # Запускаем тестирование
    test_functions()
