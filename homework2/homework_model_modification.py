import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from typing import Tuple, List, Optional
import logging
import unittest

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Вспомогательные функции и классы

def make_regression_data(n: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Генерация синтетических данных для регрессии."""
    X = torch.rand(n, 1) * 10
    y = 2 * X + 1 + torch.randn(n, 1) * 2
    return X, y


def make_classification_data(n: int = 200, n_classes: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Генерация синтетических данных для классификации."""
    if n_classes == 2:
        X = torch.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)
        return X, y
    else:
        X = torch.randn(n, 2)
        y = torch.randint(0, n_classes, (n,))
        return X, y


def log_epoch(epoch: int, loss: float, acc: Optional[float] = None):
    """Логирование метрик эпохи."""
    if acc is not None:
        logger.info(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    else:
        logger.info(f"Epoch {epoch:3d} | Loss: {loss:.4f}")


class RegressionDataset(Dataset):
    """Датасет для регрессии."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationDataset(Dataset):
    """Датасет для классификации."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Вычисление точности для бинарной классификации."""
    y_pred_class = (y_pred > 0.5).float()
    return (y_pred_class == y_true).float().mean().item()


def multi_class_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Вычисление точности для многоклассовой классификации."""
    y_pred_class = torch.argmax(y_pred, dim=1)
    return (y_pred_class == y_true).float().mean().item()



# 1.1 Модифицированная линейная регрессия

class RegularizedLinearRegression(nn.Module):
    """Линейная регрессия с L1 и L2 регуляризацией."""

    def __init__(self, in_features: int, l1_lambda: float = 0.01, l2_lambda: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def regularization_loss(self) -> torch.Tensor:
        """Вычисление L1 и L2 регуляризационных потерь."""
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


def train_linear_regression(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int = 100,
        patience: int = 5,
        verbose: bool = True
) -> List[float]:
    """Обучение модели линейной регрессии с early stopping."""
    losses = []
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)

            # Основная функция потерь + регуляризация
            loss = criterion(y_pred, batch_y)
            if hasattr(model, 'regularization_loss'):
                loss += model.regularization_loss()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

        if no_improve >= patience:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch}")
            break

    return losses


def plot_losses(losses: List[float], title: str = "Training Loss"):
    """Визуализация потерь во время обучения."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def test_linear_regression():
    """Тестирование модифицированной линейной регрессии."""
    # Генерация данных
    X, y = make_regression_data(n=200)
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Создание модели
    model = RegularizedLinearRegression(in_features=1, l1_lambda=0.01, l2_lambda=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучение
    losses = train_linear_regression(
        model, dataloader, criterion, optimizer,
        epochs=100, patience=5
    )

    # Визуализация
    plot_losses(losses, "Linear Regression with Regularization")

    # Сохранение модели
    torch.save(model.state_dict(), 'regularized_linreg.pth')

    # Загрузка модели
    loaded_model = RegularizedLinearRegression(in_features=1)
    loaded_model.load_state_dict(torch.load('regularized_linreg.pth'))
    loaded_model.eval()

    logger.info("Linear regression testing completed successfully.")



# 1.2 Модифицированная логистическая регрессия

class MultiClassLogisticRegression(nn.Module):
    """Логистическая регрессия с поддержкой многоклассовой классификации."""

    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_logistic_regression(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int = 100,
        verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """Обучение модели логистической регрессии."""
    losses = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_acc = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)

            # Для бинарной классификации нужно изменить размерность
            if model.num_classes == 1:
                loss = criterion(logits, batch_y.float())
            else:
                loss = criterion(logits, batch_y.long())

            loss.backward()
            optimizer.step()

            # Вычисление точности
            with torch.no_grad():
                if model.num_classes == 1:
                    y_pred = torch.sigmoid(logits)
                    acc = accuracy(y_pred, batch_y)
                else:
                    y_pred = torch.softmax(logits, dim=1)
                    acc = multi_class_accuracy(y_pred, batch_y)

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        losses.append(avg_loss)
        accuracies.append(avg_acc)

        if verbose and epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)

    return losses, accuracies


def evaluate_classification(
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int = 2
) -> Tuple[float, float, float, float]:
    """Вычисление метрик классификации."""
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            logits = model(batch_X)

            if num_classes == 1:
                y_prob = torch.sigmoid(logits)
                y_pred_class = (y_prob > 0.5).long()
                y_true.extend(batch_y.numpy())
                y_pred.extend(y_pred_class.numpy())
                y_score.extend(y_prob.numpy())
            else:
                y_prob = torch.softmax(logits, dim=1)
                y_pred_class = torch.argmax(y_prob, dim=1)
                y_true.extend(batch_y.numpy())
                y_pred.extend(y_pred_class.numpy())
                y_score.extend(y_prob.numpy())

    # Вычисление метрик
    precision = precision_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')
    recall = recall_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')
    f1 = f1_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')

    # ROC-AUC (только для бинарной классификации или one-vs-rest для многоклассовой)
    if num_classes == 1 or num_classes == 2:
        roc_auc = roc_auc_score(y_true, y_score if num_classes == 1 else y_score[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')

    return precision, recall, f1, roc_auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """Визуализация матрицы ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()


def test_logistic_regression():
    """Тестирование модифицированной логистической регрессии."""
    # Тестирование бинарной классификации
    X_bin, y_bin = make_classification_data(n=200, n_classes=2)
    dataset_bin = ClassificationDataset(X_bin, y_bin)
    dataloader_bin = DataLoader(dataset_bin, batch_size=32, shuffle=True)

    model_bin = MultiClassLogisticRegression(in_features=2, num_classes=1)
    criterion_bin = nn.BCEWithLogitsLoss()
    optimizer_bin = optim.SGD(model_bin.parameters(), lr=0.1)

    losses_bin, accuracies_bin = train_logistic_regression(
        model_bin, dataloader_bin, criterion_bin, optimizer_bin, epochs=100
    )

    # Визуализация
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses_bin, label='Loss')
    plt.title("Binary Classification Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies_bin, label='Accuracy', color='orange')
    plt.title("Binary Classification Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Оценка метрик
    precision, recall, f1, roc_auc = evaluate_classification(
        model_bin, dataloader_bin, num_classes=1
    )
    logger.info(f"Binary Classification Metrics:")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Матрица ошибок
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader_bin:
            logits = model_bin(batch_X)
            y_prob = torch.sigmoid(logits)
            y_pred_class = (y_prob > 0.5).long()
            y_true.extend(batch_y.numpy())
            y_pred.extend(y_pred_class.numpy())

    plot_confusion_matrix(y_true, y_pred, "Binary Classification Confusion Matrix")

    # Тестирование многоклассовой классификации
    X_multi, y_multi = make_classification_data(n=200, n_classes=3)
    dataset_multi = ClassificationDataset(X_multi, y_multi)
    dataloader_multi = DataLoader(dataset_multi, batch_size=32, shuffle=True)

    model_multi = MultiClassLogisticRegression(in_features=2, num_classes=3)
    criterion_multi = nn.CrossEntropyLoss()
    optimizer_multi = optim.SGD(model_multi.parameters(), lr=0.1)

    losses_multi, accuracies_multi = train_logistic_regression(
        model_multi, dataloader_multi, criterion_multi, optimizer_multi, epochs=100
    )

    # Визуализация
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses_multi, label='Loss')
    plt.title("Multiclass Classification Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies_multi, label='Accuracy', color='orange')
    plt.title("Multiclass Classification Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Оценка метрик
    precision, recall, f1, roc_auc = evaluate_classification(
        model_multi, dataloader_multi, num_classes=3
    )
    logger.info(f"Multiclass Classification Metrics:")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Матрица ошибок
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader_multi:
            logits = model_multi(batch_X)
            y_prob = torch.softmax(logits, dim=1)
            y_pred_class = torch.argmax(y_prob, dim=1)
            y_true.extend(batch_y.numpy())
            y_pred.extend(y_pred_class.numpy())

    plot_confusion_matrix(y_true, y_pred, "Multiclass Classification Confusion Matrix")

    # Сохранение модели
    torch.save(model_multi.state_dict(), 'multiclass_logreg.pth')

    # Загрузка модели
    loaded_model = MultiClassLogisticRegression(in_features=2, num_classes=3)
    loaded_model.load_state_dict(torch.load('multiclass_logreg.pth'))
    loaded_model.eval()

    logger.info("Logistic regression testing completed successfully.")


# Тестирование
if __name__ == '__main__':
    # Запуск тестов
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    # Демонстрация работы моделей
    logger.info("Testing Linear Regression with Regularization and Early Stopping")
    test_linear_regression()

    logger.info("\nTesting Logistic Regression with Multiclass Support")
    test_logistic_regression()
