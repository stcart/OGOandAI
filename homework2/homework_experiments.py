import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, List
import itertools
from tqdm import tqdm
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedCSVDataset(Dataset):
    """Расширенный Dataset класс с feature engineering."""

    def __init__(self, file_path: str, target_column: str,
                 degree: int = 1,
                 test_size: float = 0.2,
                 random_state: int = 42):
        super().__init__()

        self.data = pd.read_csv(file_path)
        logger.info(f"Данные загружены. Размер: {self.data.shape}")

        self.target_column = target_column
        self.degree = degree
        self.random_state = random_state

        self._preprocess_data()
        self._create_features()
        self._train_test_split(test_size)

    def _preprocess_data(self):
        """Базовая предобработка данных."""
        self.processed_data = self.data.copy()

        # Кодирование категориальных признаков
        for col in self.processed_data.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                self.processed_data[col] = LabelEncoder().fit_transform(self.processed_data[col])

        # Нормализация числовых признаков
        numeric_cols = self.processed_data.select_dtypes(include=['number']).columns.drop(self.target_column,
                                                                                          errors='ignore')
        if len(numeric_cols) > 0:
            self.scaler = StandardScaler()
            self.processed_data[numeric_cols] = self.scaler.fit_transform(self.processed_data[numeric_cols])
    def _create_features(self):
        """Создание новых признаков."""
        X = self.processed_data.drop(self.target_column, axis=1)
        y = self.processed_data[self.target_column]

        # Полиномиальные признаки
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Статистические признаки
        if self.degree >= 2:
            X_stats = pd.DataFrame()
            for col in X.select_dtypes(include=['number']).columns:
                X_stats[f'{col}_squared'] = X[col] ** 2
                X_stats[f'{col}_exp'] = np.exp(X[col])
                X_stats[f'{col}_log'] = np.log1p(np.abs(X[col]))

            X_combined = np.hstack([X_poly, X_stats])
        else:
            X_combined = X_poly

        self.features = X_combined
        self.target = y.values

        logger.info(f"Создано признаков: {X_combined.shape[1]} (исходно: {X.shape[1]})")

    def _train_test_split(self, test_size: float):
        """Разделение данных на train и test."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=self.random_state)

        logger.info(f"Данные разделены на train ({len(self.X_train)}) и test ({len(self.X_test)})")

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X_train[idx]), torch.FloatTensor([self.y_train[idx]])


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       criterion: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epochs: int = 100,
                       patience: int = 5) -> Dict:
    """Обучение и оценка модели."""
    history = {'train_loss': [], 'test_loss': []}
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Оценка на тестовых данных
        test_loss = evaluate_model(model, test_loader, criterion)
        history['test_loss'].append(test_loss)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return history


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   criterion: nn.Module) -> float:
    """Оценка модели на тестовых данных."""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            test_loss += criterion(outputs, y_batch).item()
    return test_loss / len(test_loader)


def plot_results(history: Dict, title: str):
    """Визуализация результатов обучения."""
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def hyperparameter_experiment(file_path: str,
                              target_column: str,
                              problem_type: str = 'regression'):
    """Эксперимент с различными гиперпараметрами."""
    logger.info("\n=== Исследование гиперпараметров ===")

    # Параметры для экспериментов
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop
    }

    dataset = EnhancedCSVDataset(file_path, target_column, degree=1)
    input_dim = dataset.X_train.shape[1]

    # Создаем DataLoader для тестовых данных
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(dataset.X_test),
        torch.FloatTensor(dataset.y_test.reshape(-1, 1))
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    results = []

    for lr, batch_size, (opt_name, opt_class) in tqdm(
            itertools.product(learning_rates, batch_sizes, optimizers.items()),
            total=len(learning_rates) * len(batch_sizes) * len(optimizers),
            desc="Running experiments"
    ):
        # Создаем DataLoader с текущим batch_size
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset.X_train),
            torch.FloatTensor(dataset.y_train.reshape(-1, 1))
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Создаем модель
        if problem_type == 'regression':
            model = nn.Linear(input_dim, 1)
            criterion = nn.MSELoss()
        else:
            model = nn.Linear(input_dim, 1)
            criterion = nn.BCEWithLogitsLoss()

        # Создаем оптимизатор
        optimizer = opt_class(model.parameters(), lr=lr)

        # Обучение и оценка
        history = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, epochs=50)

        # Сохраняем результаты
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer': opt_name,
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'best_test_loss': min(history['test_loss'])
        })

    # Визуализация результатов
    results_df = pd.DataFrame(results)
    print("\nРезультаты экспериментов:")
    print(results_df.sort_values('best_test_loss').head())

    # Графики для лучших комбинаций
    for param in ['learning_rate', 'batch_size', 'optimizer']:
        plt.figure(figsize=(10, 5))
        for value in results_df[param].unique():
            subset = results_df[results_df[param] == value]
            plt.plot(subset['best_test_loss'], 'o-', label=f'{param}={value}')
        plt.title(f"Влияние {param} на качество модели")
        plt.xlabel('Эксперимент')
        plt.ylabel('Test Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    return results_df


def feature_engineering_experiment(file_path: str,
                                   target_column: str,
                                   problem_type: str = 'regression'):
    """Эксперимент с feature engineering."""
    logger.info("\n=== Feature Engineering ===")

    results = []

    for degree in [1, 2, 3]:
        dataset = EnhancedCSVDataset(file_path, target_column, degree=degree)
        input_dim = dataset.X_train.shape[1]

        # Создаем DataLoader'ы
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset.X_train),
            torch.FloatTensor(dataset.y_train.reshape(-1, 1))
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset.X_test),
            torch.FloatTensor(dataset.y_test.reshape(-1, 1))
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Создаем модель
        if problem_type == 'regression':
            model = nn.Linear(input_dim, 1)
            criterion = nn.MSELoss()
        else:
            model = nn.Linear(input_dim, 1)
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Обучение и оценка
        history = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, epochs=50)

        # Сохраняем результаты
        results.append({
            'degree': degree,
            'num_features': input_dim,
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'best_test_loss': min(history['test_loss'])
        })

        # Визуализация обучения
        plot_results(history, f"Polynomial Features (degree={degree})")

    # Сравнение результатов
    results_df = pd.DataFrame(results)
    print("\nСравнение feature engineering подходов:")
    print(results_df)

    return results_df


if __name__ == '__main__':
    # Пример использования

    # 1. Эксперименты с гиперпараметрами (регрессия)
    hyperparameter_experiment('test.csv', 'y', problem_type='regression')

    # 2. Feature engineering (классификация)
    feature_engineering_experiment('Social_Network_Ads.csv', 'Purchased', problem_type='classification')
