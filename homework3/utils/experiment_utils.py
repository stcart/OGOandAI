import torch
import time
import logging
from typing import Dict, List
from utils.model_utils import FullyConnectedModel, count_parameters
from utils.visualization_utils import save_plot
from datasets import get_mnist_loaders, get_cifar_loaders
from trainer import train_model

MODEL_CONFIGS = {
    '1_layer': {
        'input_size': 784,
        'num_classes': 10,
        'layers': []
    },
    '2_layers': {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'}
        ]
    },
    '3_layers': {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'}
        ]
    },
    '5_layers': {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'},
            {'type': 'linear', 'size': 64},
            {'type': 'relu'}
        ]
    },
    '7_layers': {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'},
            {'type': 'linear', 'size': 64},
            {'type': 'relu'},
            {'type': 'linear', 'size': 32},
            {'type': 'relu'},
            {'type': 'linear', 'size': 16},
            {'type': 'relu'}
        ]
    },
    '5_layers_with_dropout': {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'},
            {'type': 'batch_norm'},
            {'type': 'dropout', 'rate': 0.2},
            {'type': 'linear', 'size': 64},
            {'type': 'relu'}
        ]
    }
}

def create_and_train_model(model_config: Dict, model_name: str,
                         train_loader, test_loader,
                         epochs: int = 10, lr: float = 0.001,
                         device: str = 'cpu') -> Dict:
    """Создает и обучает модель с заданной конфигурацией"""
    logging.info(f"Создание и обучение модели: {model_name}")
    model = FullyConnectedModel(**model_config).to(device)
    logging.info(f"Количество параметров модели: {count_parameters(model)}")

    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    training_time = time.time() - start_time

    results = {
        'model_name': model_name,
        'history': history,
        'training_time': training_time,
        'num_params': count_parameters(model),
        'final_train_acc': history['train_accs'][-1],
        'final_test_acc': history['test_accs'][-1]
    }

    logging.info(f"Обучение завершено. Время обучения: {training_time:.2f} сек")
    logging.info(f"Точность на обучающих данных: {results['final_train_acc']:.4f}")
    logging.info(f"Точность на тестовых данных: {results['final_test_acc']:.4f}")
    return results

def compare_model_depths(dataset_name: str = 'mnist', epochs: int = 15,
                       batch_size: int = 64, lr: float = 0.001) -> List[Dict]:
    """Сравнивает модели разной глубины на заданном датасете"""
    logging.info(f"Начало эксперимента: сравнение глубины моделей на {dataset_name}")

    if dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_size = 784
    elif dataset_name == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_size = 3072
    else:
        raise ValueError("Поддерживаются только 'mnist' и 'cifar'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Используемое устройство: {device}")

    configs = MODEL_CONFIGS.copy()
    for config in configs.values():
        config['input_size'] = input_size

    results = []
    for model_name in ['1_layer', '2_layers', '3_layers', '5_layers', '7_layers']:
        model_config = configs[model_name]
        result = create_and_train_model(
            model_config, model_name, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        results.append(result)

    return results

def analyze_overfitting(dataset_name: str = 'mnist', epochs: int = 20,
                      batch_size: int = 64, lr: float = 0.001) -> Dict:
    """Анализирует влияние глубины сети на переобучение"""
    logging.info(f"Начало эксперимента: анализ переобучения на {dataset_name}")

    if dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_size = 784
    elif dataset_name == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_size = 3072
    else:
        raise ValueError("Поддерживаются только 'mnist' и 'cifar'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = MODEL_CONFIGS.copy()
    for config in configs.values():
        config['input_size'] = input_size

    logging.info("Обучение модели без регуляризации (5 слоев)")
    model_5l_config = configs['5_layers']
    result_5l = create_and_train_model(
        model_5l_config, '5_layers', train_loader, test_loader,
        epochs=epochs, lr=lr, device=device
    )

    logging.info("Обучение модели с регуляризацией (5 слоев + Dropout + BatchNorm)")
    model_reg_config = configs['5_layers_with_dropout']
    result_reg = create_and_train_model(
        model_reg_config, '5_layers_with_dropout', train_loader, test_loader,
        epochs=epochs, lr=lr, device=device
    )

    return {'without_regularization': result_5l, 'with_regularization': result_reg}
