import torch.nn as nn
import torch

class FullyConnectedModel(nn.Module):
    def __init__(self, config=None, input_size=None, num_classes=None, **kwargs):
        super().__init__()
        # ... (исходный код класса)

    def _build_layers(self):
        # ... (исходный код метода)
    
    def forward(self, x):
        # ... (исходный код метода)

def count_parameters(model):
    """Подсчет параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    """Сохранение модели"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загрузка модели"""
    model.load_state_dict(torch.load(path))
    return model
