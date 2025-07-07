import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from tqdm import tqdm

def run_experiment(dataset_path, sizes=[64, 128, 224, 512], num_images=100):
    """Проводит эксперимент с разными размерами изображений"""
    # Результаты
    time_results = []
    memory_results = []
    
    for size in sizes:
        print(f"\nЭксперимент с размером {size}x{size}...")
        
        # Создаем датасет с нужным размером
        dataset = CustomImageDataset(
            dataset_path,
            transform=transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()
            ]),
            target_size=(size, size)
        )
        
        # Выбираем первые num_images изображений
        indices = range(min(num_images, len(dataset)))
        
        # Измеряем время и память
        tracemalloc.start()
        start_time = time.time()
        
        # Загрузка и обработка изображений
        for i in tqdm(indices, desc=f"Обработка {size}x{size}"):
            img, _ = dataset[i]
            # Применяем стандартные аугментации
            aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(15)
            ])
            _ = aug(img)
        
        # Замер показателей
        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        time_results.append(elapsed_time)
        memory_results.append(peak / 1024**2)  # в MB
        
        print(f"Время: {elapsed_time:.2f} сек")
        print(f"Пиковое использование памяти: {peak / 1024**2:.2f} MB")
    
    return sizes, time_results, memory_results

def plot_results(sizes, times, memories):
    """Визуализирует результаты эксперимента"""
    plt.figure(figsize=(12, 5))
    
    # График времени
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, 'bo-')
    plt.title('Зависимость времени от размера изображения')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время обработки (сек)')
    plt.grid(True)
    
    # График памяти
    plt.subplot(1, 2, 2)
    plt.plot(sizes, memories, 'ro-')
    plt.title('Зависимость памяти от размера изображения')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Пиковое использование памяти (MB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'size_experiment.jpg'))
    plt.show()

# Запуск эксперимента
if __name__ == "__main__":
    sizes = [64, 128, 224, 512]  # Тестируемые размеры
    num_images = 100  # Количество изображений для теста
    
    print(f"Начинаем эксперимент с {num_images} изображениями...")
    sizes, times, memories = run_experiment(TRAIN_PATH, sizes, num_images)
    
    print("\nРезультаты:")
    for size, time, mem in zip(sizes, times, memories):
        print(f"{size}x{size}: {time:.2f} сек, {mem:.2f} MB")
    
    plot_results(sizes, times, memories)
    print(f"\nГрафики сохранены в {RESULTS_PATH}/size_experiment.jpg")
