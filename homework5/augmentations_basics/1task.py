import os
import zipfile
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# КОНФИГУРАЦИЯ ПУТЕЙ
ZIP_PATH = "/content/dataset-20250707T173044Z-1-001.zip"  # Путь к архиву
EXTRACT_ROOT = "/content"                                # Куда распаковываем 
TRAIN_PATH = os.path.join(EXTRACT_ROOT, "data/train")    # Новый путь к train
RESULTS_PATH = "/content/results"                        # Для сохранения результатов

# РАСПАКОВКА ДАННЫХ
if not os.path.exists(TRAIN_PATH):
    print("Распаковываем архив...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_ROOT)  # Распаковываем в /content 
    print(f"Данные распакованы в {EXTRACT_ROOT}")

# Проверяем новую структуру
if not os.path.exists(TRAIN_PATH):
    available = [f for f in os.listdir(EXTRACT_ROOT) if os.path.isdir(os.path.join(EXTRACT_ROOT, f))]
    raise FileNotFoundError(
        f"Ошибка: папка {TRAIN_PATH} не найдена после распаковки!\n"
        f"Доступные папки в {EXTRACT_ROOT}: {available}"
    )

print(f"\nСтруктура данных:")
print(f"Расположение train: {TRAIN_PATH}")
print(f"Содержимое train:", os.listdir(TRAIN_PATH)[:5], "...\n")  # Покажем первые 5 классов

# ЗАГРУЖАЕМ ФАЙЛЫ
import sys
sys.path.append('/content')

try:
    from custom_datasets import CustomImageDataset
    from extra_augs import (AddGaussianNoise, RandomErasingCustom, 
                           CutOut, Solarize, Posterize, AutoContrast)
    from utils import show_images, show_single_augmentation, show_multiple_augmentations
except ImportError as e:
    raise ImportError(
        f"Ошибка загрузки модулей! Убедитесь что файлы находятся в /content:\n"
        f"Доступные файлы: {os.listdir('/content')}\n"
        f"Ошибка: {e}"
    )

# ОСНОВНОЙ КОД С АУГМЕНТАЦИЯМИ
def main():
    # Создаем папку для результатов
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Загружаем датасет
    dataset = CustomImageDataset(TRAIN_PATH, transform=None, target_size=(224, 224))
    
    # Выбираем по 1 изображению из первых 5 классов
    samples = []
    current_class = -1
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label != current_class and label < 5:
            samples.append((img, label))
            current_class = label
            if len(samples) == 5:
                break

    # Определяем аугментации
    standard_augs = [
        ("Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),
        ("Random Crop", transforms.RandomCrop(200, padding=20)),
        ("Color Jitter", transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)),
        ("Rotation 30°", transforms.RandomRotation(30)),
        ("Grayscale", transforms.RandomGrayscale(p=1.0))
    ]

    # Обработка каждого изображения
    for original_img, label in samples:
        class_name = dataset.get_class_names()[label]
        print(f"\nОбработка класса: {class_name}")

        # Сохраняем оригинал
        original_img.save(f"{RESULTS_PATH}/original_{class_name}.jpg")

        # Применяем все аугментации по очереди
        augmented_images = []
        for name, aug in standard_augs:
            transform = transforms.Compose([aug, transforms.ToTensor()])
            aug_img = transform(original_img)
            
            # Сохраняем каждую аугментацию
            plt.figure()
            show_single_augmentation(transforms.ToTensor()(original_img), aug_img, name)
            plt.savefig(f"{RESULTS_PATH}/{class_name}_{name.replace(' ', '_')}.jpg")
            plt.close()
            
            augmented_images.append(aug_img)

        # Сохраняем все аугментации вместе
        show_multiple_augmentations(
            transforms.ToTensor()(original_img),
            augmented_images,
            [name for name, _ in standard_augs]
        )
        plt.savefig(f"{RESULTS_PATH}/{class_name}_all_augs.jpg")
        plt.close()

    print(f"\nГотово! Результаты сохранены в {RESULTS_PATH}")

if __name__ == "__main__":
    main()
