import matplotlib.pyplot as plt
import os

def save_plot(plt, filename, subfolder="depth_experiments"):
    """Сохраняет график в results/"""
    path = f"results/{subfolder}/{filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def plot_comparison(results: List[Dict], metric: str = 'acc') -> None:
    """Сравнение моделей"""
    # ... (ваш код функции, добавить save_plot в конце)
    save_plot(plt, f"{metric}_comparison.png", "depth_experiments")

def plot_overfitting_analysis(results: Dict) -> None:
    """Анализ переобучения"""
    # ... (ваш код функции, добавить save_plot в конце)
    save_plot(plt, "overfitting_analysis.png", "regularization_experiments")
