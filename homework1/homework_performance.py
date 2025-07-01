import torch
import time
from tabulate import tabulate


def create_matrices():
    """
    Создание больших матриц для тестирования (задание 3.1)

    Returns:
        dict: Словарь с матрицами разных размеров:
            - '64x1024x1024': матрица 64x1024x1024
            - '128x512x512': матрица 128x512x512
            - '256x256x256': матрица 256x256x256
    """
    # Создаем матрицы разных размеров
    matrices = {
        '64x1024x1024': torch.rand(64, 1024, 1024),
        '128x512x512': torch.rand(128, 512, 512),
        '256x256x256': torch.rand(256, 256, 256)
    }
    return matrices


def measure_time(device, operation, *args):
    """
    Измерение времени выполнения операции (задание 3.2)

    Args:
        device (str): 'cpu' или 'cuda'
        operation (function): Функция, выполняющая операцию
        *args: Аргументы для операции

    Returns:
        float: Время выполнения в миллисекундах
    """
    # Синхронизируем для точного измерения
    if device == 'cuda':
        torch.cuda.synchronize()

    if device == 'cuda':
        # Используем CUDA events для точного измерения на GPU
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        operation(*args)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
    else:
        # Используем time.time() для CPU
        start_time = time.time()
        operation(*args)
        elapsed_time = (time.time() - start_time) * 1000  # в миллисекунды

    return elapsed_time


def compare_operations(matrices):
    """
    Сравнение производительности операций на CPU и GPU (задание 3.3)

    Args:
        matrices (dict): Словарь с матрицами для тестирования

    Returns:
        dict: Результаты сравнения для каждой матрицы
    """
    results = {}

    for name, matrix in matrices.items():
        print(f"\n=== Тестирование матрицы {name} ===")

        # Копируем матрицу на CPU и GPU (если доступно)
        cpu_matrix = matrix.clone()
        gpu_matrix = matrix.clone().cuda() if torch.cuda.is_available() else None

        # Операции для тестирования
        operations = {
            'Матричное умножение': lambda x: torch.matmul(x, x.transpose(-2, -1)),
            'Поэлементное сложение': lambda x: x + x,
            'Поэлементное умножение': lambda x: x * x,
            'Транспонирование': lambda x: x.transpose(-2, -1),
            'Сумма всех элементов': lambda x: torch.sum(x)
        }

        table = []

        for op_name, op_func in operations.items():
            # Измеряем время на CPU
            cpu_time = measure_time('cpu', op_func, cpu_matrix)

            # Измеряем время на GPU
            gpu_time = None
            speedup = None

            if gpu_matrix is not None:
                try:
                    gpu_time = measure_time('cuda', op_func, gpu_matrix)
                    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                except RuntimeError as e:
                    print(f"Ошибка при выполнении {op_name} на GPU: {e}")
                    gpu_time = None

            # Добавляем результаты в таблицу
            row = [op_name]
            row.append(f"{cpu_time:.1f}")
            row.append(f"{gpu_time:.1f}" if gpu_time is not None else "N/A")
            row.append(f"{speedup:.1f}x" if speedup is not None else "N/A")
            table.append(row)

        # Выводим результаты в табличном виде
        headers = ["Операция", "CPU (мс)", "GPU (мс)", "Ускорение"]
        print(tabulate(table, headers=headers, tablefmt="grid"))

        results[name] = table

    return results


def analyze_results(results):
    """
    Анализ результатов сравнения (задание 3.4)
    """
    print("\n=== Анализ результатов ===")

    if not torch.cuda.is_available():
        print("GPU не доступен. Анализ невозможен.")
        return

    print("\n1. Какие операции получают наибольшее ускорение на GPU?")
    print("Матричные операции (умножение, сложение) обычно получают наибольшее ускорение,")
    print("так как GPU оптимизирован для параллельного выполнения таких операций.")

    print("\n2. Почему некоторые операции могут быть медленнее на GPU?")
    print("Простые операции (например, сумма) могут быть медленнее из-за накладных расходов")
    print("на передачу данных и запуск ядер GPU. Для маленьких матриц CPU может быть быстрее.")

    print("\n3. Как размер матриц влияет на ускорение?")
    print("Чем больше матрицы, тем больше выигрыш от GPU. Для маленьких матриц")
    print("накладные расходы могут превысить выгоду от параллелизации.")

    print("\n4. Что происходит при передаче данных между CPU и GPU?")
    print("Передача данных через PCIe шину может стать узким местом.")
    print("Оптимально минимизировать передачу данных между устройствами.")


def main():
    # Проверяем доступность CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступен: {cuda_available}")
    if cuda_available:
        print(f"Устройство: {torch.cuda.get_device_name(0)}")

    # 3.1 Создаем матрицы для тестирования
    matrices = create_matrices()

    # 3.3 Сравниваем производительность операций
    results = compare_operations(matrices)

    # 3.4 Анализируем результаты
    analyze_results(results)


if __name__ == "__main__":
    main()
