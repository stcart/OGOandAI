import torch
import time
import logging
from typing import Dict, List
from utils.model_utils import FullyConnectedModel, count_parameters
from utils.visualization_utils import save_plot
from datasets import get_mnist_loaders, get_cifar_loaders
from trainer import train_model

MODEL_CONFIGS = {
    '1_layer': {'input_size': 784, 'num_classes': 10, 'layers': []},
    '2_layers': {'input_size': 784, 'num_classes': 10, 'layers': [
        {'type': 'linear', 'size': 512}, {'type': 'relu'}]},
    # ... (остальные конфиги из вашего кода)
}

def create_and_train_model(model_config: Dict, model_name: str,
                         train_loader, test_loader,
                         epochs: int = 10, lr: float = 0.001,
                         device: str = 'cpu') -> Dict:
    """Создает и обучает модель"""
    # ... (ваш код функции без изменений)
    return results

def compare_model_depths(dataset_name: str = 'mnist', epochs: int = 15,
                       batch_size: int = 64, lr: float = 0.001) -> List[Dict]:
    """Сравнивает модели разной глубины"""
    # ... (ваш код функции без изменений)
    return results

def analyze_overfitting(dataset_name: str = 'mnist', epochs: int = 20,
                      batch_size: int = 64, lr: float = 0.001) -> Dict:
    """Анализирует переобучение"""
    # ... (ваш код функции без изменений)
    return results
