import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

def analyze_dataset(dataset_path):
    # Подсчет количества изображений по классам
    class_counts = defaultdict(int)
    class_sizes = defaultdict(list)
    all_sizes = []
    
    print("Анализ датасета...")
    
    # Проходим по всем изображениям
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    class_counts[class_name] += 1
                    class_sizes[class_name].append((width, height))
                    all_sizes.append((width, height))
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
    
    # Статистика по размерам
    widths, heights = zip(*all_sizes)
    min_size = (min(widths), min(heights))
    max_size = (max(widths), max(heights))
    avg_size = (np.mean(widths), np.mean(heights))
    
    # Визуализация
    plt.figure(figsize=(18, 6))
    
    # Гистограмма распределения классов
    plt.subplot(1, 3, 1)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Количество изображений по классам')
    plt.xlabel('Классы')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Распределение размеров (scatter plot)
    plt.subplot(1, 3, 2)
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Распределение размеров изображений')
    plt.xlabel('Ширина (px)')
    plt.ylabel('Высота (px)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Гистограмма соотношения сторон
    plt.subplot(1, 3, 3)
    ratios = [w/h for w, h in all_sizes]
    plt.hist(ratios, bins=30, edgecolor='black')
    plt.title('Распределение соотношений сторон (width/height)')
    plt.xlabel('Соотношение сторон')
    plt.ylabel('Количество')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'dataset_analysis.jpg'))
    plt.show()
    
    # Вывод статистики
    print("\nСтатистика датасета:")
    print(f"Всего классов: {len(class_counts)}")
    print(f"Всего изображений: {sum(class_counts.values())}")
    print(f"\nРазмеры изображений:")
    print(f"Минимальный: {min_size[0]}x{min_size[1]} px")
    print(f"Максимальный: {max_size[0]}x{max_size[1]} px")
    print(f"Средний: {avg_size[0]:.1f}x{avg_size[1]:.1f} px")
    
    print("\nКоличество изображений по классам:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {count} изображений")
    
    return class_counts, class_sizes

# Запуск анализа
if __name__ == "__main__":
    class_counts, class_sizes = analyze_dataset(TRAIN_PATH)
    print(f"\nРезультаты сохранены в {RESULTS_PATH}/dataset_analysis.jpg")
