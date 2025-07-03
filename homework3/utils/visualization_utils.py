import matplotlib.pyplot as plt
import os

def plot_comparison(results: List[Dict], metric: str = 'acc') -> None:
    """
    Визуализирует сравнение моделей по заданной метрике.

    Args:
        results: Список с результатами обучения моделей
        metric: Метрика для сравнения ('acc' или 'loss')
    """
    model_names = [res['model_name'] for res in results]

    if metric == 'acc':
        train_metrics = [res['final_train_acc'] for res in results]
        test_metrics = [res['final_test_acc'] for res in results]
        ylabel = 'Accuracy'
    else:
        train_metrics = [res['history']['train_losses'][-1] for res in results]
        test_metrics = [res['history']['test_losses'][-1] for res in results]
        ylabel = 'Loss'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График точности/потерь
    ax1.plot(model_names, train_metrics, 'o-', label='Train')
    ax1.plot(model_names, test_metrics, 'o-', label='Test')
    ax1.set_title(f'Model Comparison ({ylabel})')
    ax1.set_xlabel('Model')
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)

    # График времени обучения и количества параметров
    times = [res['training_time'] for res in results]
    params = [res['num_params'] for res in results]

    ax2.plot(model_names, times, 'o-', label='Training Time (s)')
    ax2.set_title('Training Time and Model Size')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)

    ax2b = ax2.twinx()
    ax2b.plot(model_names, params, 'o-r', label='Number of Parameters')
    ax2b.set_ylabel('Number of Parameters', color='r')
    ax2b.tick_params(axis='y', labelcolor='r')
    ax2b.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_overfitting_analysis(results: Dict) -> None:
    """
    Визуализирует анализ переобучения.

    Args:
        results: Результаты обучения моделей с регуляризацией и без
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Графики для модели без регуляризации
    res_no_reg = results['without_regularization']
    ax1.plot(res_no_reg['history']['train_accs'], label='Train Acc (no reg)')
    ax1.plot(res_no_reg['history']['test_accs'], label='Test Acc (no reg)')
    ax1.set_title('Model without Regularization')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Графики для модели с регуляризацией
    res_reg = results['with_regularization']
    ax2.plot(res_reg['history']['train_accs'], label='Train Acc (with reg)')
    ax2.plot(res_reg['history']['test_accs'], label='Test Acc (with reg)')
    ax2.set_title('Model with Regularization')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
