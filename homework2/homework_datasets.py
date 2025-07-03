import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Union, List
import unittest
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSVDataset(Dataset):
    """Кастомный Dataset класс для работы с CSV файлами."""

    def __init__(self, file_path: str, target_column: str,
                 normalize_numeric: bool = True,
                 test_size: float = 0.2,
                 random_state: int = 42):
        super().__init__()

        self.data = pd.read_csv(file_path)
        logger.info(f"Данные загружены. Размер: {self.data.shape}")

        self.target_column = target_column
        self.normalize_numeric = normalize_numeric
        self.random_state = random_state

        self._identify_feature_types()
        self._preprocess_data()
        self._train_test_split(test_size)

    def _identify_feature_types(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.binary_cols = []

        for col in self.data.columns:
            if col == self.target_column:
                continue
            if self.data[col].nunique() == 2:
                self.binary_cols.append(col)
            elif self.data[col].dtype == 'object':
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)

        logger.info(
            f"Типы признаков: numeric={self.numeric_cols}, categorical={self.categorical_cols}, binary={self.binary_cols}")

    def _preprocess_data(self):
        self.processed_data = self.data.copy()

        for col in self.categorical_cols:
            self.processed_data[col] = LabelEncoder().fit_transform(self.processed_data[col])

        for col in self.binary_cols:
            self.processed_data[col] = LabelEncoder().fit_transform(self.processed_data[col])

        if self.normalize_numeric and self.numeric_cols:
            self.scaler = StandardScaler()
            self.processed_data[self.numeric_cols] = self.scaler.fit_transform(
                self.processed_data[self.numeric_cols])
            logger.info("Числовые признаки нормализованы")

        if self.data[self.target_column].nunique() == 2:
            self.problem_type = 'classification'
            self.processed_data[self.target_column] = LabelEncoder().fit_transform(
                self.processed_data[self.target_column])
            logger.info("Бинарная классификация")
        else:
            self.problem_type = 'regression'
            logger.info("Регрессия")

    def _train_test_split(self, test_size: float):
        features = self.processed_data.drop(self.target_column, axis=1)
        target = self.processed_data[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features.values, target.values, test_size=test_size, random_state=self.random_state)

        logger.info(f"Данные разделены на train ({len(self.X_train)}) и test ({len(self.X_test)})")

    def get_feature_dim(self) -> int:
        return self.X_train.shape[1]

    def get_problem_type(self) -> str:
        return self.problem_type


class CSVDataLoader:
    def __init__(self, dataset: CSVDataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size

        X_train = torch.FloatTensor(dataset.X_train)
        X_test = torch.FloatTensor(dataset.X_test)

        if dataset.problem_type == 'regression':
            y_train = torch.FloatTensor(dataset.y_train).view(-1, 1)
            y_test = torch.FloatTensor(dataset.y_test).view(-1, 1)
        else:
            y_train = torch.FloatTensor(dataset.y_train).view(-1, 1)
            y_test = torch.FloatTensor(dataset.y_test).view(-1, 1)

        self.train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        self.test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        def collate_fn(batch):
            X = torch.stack([item[0] for item in batch])
            y = torch.stack([item[1] for item in batch])
            return X, y

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, epochs: int = 100, patience: int = 5):
    model.train()
    history = {'train_loss': []}
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)

        avg_loss = total_loss / total_samples
        history['train_loss'].append(avg_loss)

        logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)

    return total_loss / total_samples


def plot_history(history: Dict[str, List[float]], title: str = "Training History"):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_regression_experiment(file_path: str, target_column: str):
    logger.info(f"\n=== Регрессионный эксперимент ===")
    logger.info(f"Датасет: {file_path}, целевая переменная: {target_column}")

    dataset = CSVDataset(file_path, target_column)
    data_loader = CSVDataLoader(dataset)
    train_loader, test_loader = data_loader.get_loaders()

    model = nn.Linear(dataset.get_feature_dim(), 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    history = train_model(model, train_loader, criterion, optimizer, epochs=100)
    test_loss = evaluate_model(model, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Сохраняем модель
    model_path = 'linear_regression_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Модель линейной регрессии сохранена в {model_path}")

    # Сохраняем scaler для нормализации новых данных
    if hasattr(dataset, 'scaler'):
        scaler_path = 'linear_regression_scaler.pkl'
        import joblib
        joblib.dump(dataset.scaler, scaler_path)
        logger.info(f"Scaler сохранен в {scaler_path}")

    plot_history(history, "Linear Regression Training Loss")
    return model, history, test_loss


def run_classification_experiment(file_path: str, target_column: str):
    logger.info(f"\n=== Классификационный эксперимент ===")
    logger.info(f"Датасет: {file_path}, целевая переменная: {target_column}")

    dataset = CSVDataset(file_path, target_column)
    data_loader = CSVDataLoader(dataset)
    train_loader, test_loader = data_loader.get_loaders()

    model = nn.Linear(dataset.get_feature_dim(), 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    history = train_model(model, train_loader, criterion, optimizer, epochs=100)
    test_loss = evaluate_model(model, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Сохраняем модель
    model_path = 'logistic_regression_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Модель логистической регрессии сохранена в {model_path}")

    # Сохраняем scaler для нормализации новых данных
    if hasattr(dataset, 'scaler'):
        scaler_path = 'logistic_regression_scaler.pkl'
        import joblib
        joblib.dump(dataset.scaler, scaler_path)
        logger.info(f"Scaler сохранен в {scaler_path}")

    plot_history(history, "Logistic Regression Training Loss")
    return model, history, test_loss


def load_regression_model():
    """Загрузка сохраненной модели линейной регрессии"""
    model = nn.Linear(1, 1)  # Создаем модель такой же архитектуры
    model.load_state_dict(torch.load('linear_regression_model.pth'))
    model.eval()
    return model


def load_classification_model(input_dim):
    """Загрузка сохраненной модели логистической регрессии"""
    model = nn.Linear(input_dim, 1)  # Создаем модель такой же архитектуры
    model.load_state_dict(torch.load('logistic_regression_model.pth'))
    model.eval()
    return model

# Тестирование
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    # Получаем путь к директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Эксперимент с регрессией
    test_csv_path = os.path.join(script_dir, 'test.csv')
    if os.path.exists(test_csv_path):
        run_regression_experiment(test_csv_path, 'y')
    else:
        logger.error(f"Файл {test_csv_path} не найден!")

    # Эксперимент с классификацией
    social_ads_path = os.path.join(script_dir, 'Social_Network_Ads.csv')
    if os.path.exists(social_ads_path):
        run_classification_experiment(social_ads_path, 'Purchased')
    else:
        logger.error(f"Файл {social_ads_path} не найден!")
