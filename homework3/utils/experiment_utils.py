import torch
import time
import os
import json
from tqdm import tqdm
from torch import optim
from utils.model_utils import FullyConnectedModel, count_parameters
from utils.visualization_utils import save_plot

def analyze_overfitting(dataset_name: str = 'mnist', epochs: int = 20,
                        batch_size: int = 64, lr: float = 0.001) -> Dict:
    """
    Анализирует влияние глубины сети на переобучение.

    Args:
        dataset_name: Имя датасета ('mnist' или 'cifar')
        epochs: Количество эпох обучения
        batch_size: Размер батча
        lr: Скорость обучения

    Returns:
        Результаты для модели с регуляризацией и без
    """
    logger.info(f"Начало эксперимента: анализ переобучения на {dataset_name}")

    # Загрузка данных
    if dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_size = 784
    elif dataset_name == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_size = 3072  # 32x32x3
    else:
        raise ValueError("Поддерживаются только 'mnist' и 'cifar'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Модификация конфигураций под выбранный датасет
    configs = MODEL_CONFIGS.copy()
    for config in configs.values():
        config['input_size'] = input_size

    # Обучение модели без регуляризации
    logger.info("Обучение модели без регуляризации (5 слоев)")
    model_5l_config = configs['5_layers']
    result_5l = create_and_train_model(
        model_5l_config, '5_layers', train_loader, test_loader,
        epochs=epochs, lr=lr, device=device
    )

    # Обучение модели с регуляризацией
    logger.info("Обучение модели с регуляризацией (5 слоев + Dropout + BatchNorm)")
    model_reg_config = configs['5_layers_with_dropout']
    result_reg = create_and_train_model(
        model_reg_config, '5_layers_with_dropout', train_loader, test_loader,
        epochs=epochs, lr=lr, device=device
    )

    return {'without_regularization': result_5l, 'with_regularization': result_reg}



def compare_model_depths(dataset_name: str = 'mnist', epochs: int = 15,
                         batch_size: int = 64, lr: float = 0.001) -> List[Dict]:
    """
    Сравнивает модели разной глубины на заданном датасете.

    Args:
        dataset_name: Имя датасета ('mnist' или 'cifar')
        epochs: Количество эпох обучения
        batch_size: Размер батча
        lr: Скорость обучения

    Returns:
        Список с результатами для всех моделей
    """
    logger.info(f"Начало эксперимента: сравнение глубины моделей на {dataset_name}")

    # Загрузка данных
    if dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_size = 784
    elif dataset_name == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_size = 3072  # 32x32x3
    else:
        raise ValueError("Поддерживаются только 'mnist' и 'cifar'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Используемое устройство: {device}")

    # Модификация конфигураций под выбранный датасет
    configs = MODEL_CONFIGS.copy()
    for config in configs.values():
        config['input_size'] = input_size

    # Обучение моделей разной глубины
    results = []
    for model_name in ['1_layer', '2_layers', '3_layers', '5_layers', '7_layers']:
        model_config = configs[model_name]
        result = create_and_train_model(
            model_config, model_name, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        results.append(result)

    return results
