import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Сначала определим класс CustomImageDataset прямо в ноутбуке
class CustomImageDataset(torch.utils.data.Dataset):
    """Кастомный датасет для работы с папками классов"""
    
    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            transform: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Собираем все пути к изображениям
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        
        # Ресайзим изображение
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Применяем аугментации
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes

# Функции для визуализации
def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def setup_environment():
    """Настраивает окружение: распаковывает архив и создает папки"""
    # Создаем папку для данных, если ее нет
    os.makedirs('data', exist_ok=True)
    
    # Проверяем, есть ли уже распакованные данные
    if not os.path.exists('data/train'):
        # Загружаем архив (в Colab нужно загрузить вручную через файловый менеджер)
        zip_path = 'dataset-20250707T173044Z-1-001.zip'
        
        if os.path.exists(zip_path):
            print("Распаковываем архив...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('data')
            print("Распаковка завершена!")
        else:
            raise FileNotFoundError(f"Архив {zip_path} не найден. Пожалуйста, загрузите его в корневую директорию.")
    
    # Создаем папку для результатов
    os.makedirs('results', exist_ok=True)

def main():
    # Настраиваем окружение
    setup_environment()
    
    # Загрузка датасета без аугментаций
    root = 'data/train'
    dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
    
    # Выбираем по одному изображению из первых 5 классов
    sample_indices = []
    current_class = -1
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label != current_class and label < 5:  # Берем только первые 5 классов
            sample_indices.append(idx)
            current_class = label
            if len(sample_indices) == 5:
                break
    
    # Определяем стандартные аугментации
    standard_augs = [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomCrop", transforms.RandomCrop(200, padding=20)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
        ("RandomRotation", transforms.RandomRotation(degrees=30)),
        ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
    ]
    
    # Применяем аугментации к каждому изображению
    for img_idx in sample_indices:
        original_img, label = dataset[img_idx]
        class_name = dataset.get_class_names()[label]
        
        # Визуализация оригинального изображения
        plt.figure(figsize=(5, 5))
        plt.imshow(original_img)
        plt.title(f"Оригинал: {class_name}")
        plt.axis('off')
        plt.savefig(f'results/original_{class_name}.png', bbox_inches='tight')
        plt.close()
        
        # Визуализация каждой аугментации отдельно
        augmented_imgs = []
        titles = []
        
        for name, aug in standard_augs:
            # Создаем трансформацию с одной аугментацией
            aug_transform = transforms.Compose([
                aug,
                transforms.ToTensor()
            ])
            aug_img = aug_transform(original_img)
            
            # Сохраняем для группового отображения
            augmented_imgs.append(aug_img)
            titles.append(name)
            
            # Визуализация отдельной аугментации
            show_single_augmentation(
                transforms.ToTensor()(original_img),
                aug_img,
                title=name
            )
            plt.savefig(f'results/{class_name}_{name}.png', bbox_inches='tight')
            plt.close()
        
        # Визуализация всех аугментаций вместе
        show_multiple_augmentations(
            transforms.ToTensor()(original_img),
            augmented_imgs,
            titles
        )
        plt.savefig(f'results/{class_name}_all_augs.png', bbox_inches='tight')
        plt.close()
        
        # Применение всех аугментаций вместе
        combined_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomCrop(200, padding=20),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor()
        ])
        combined_img = combined_aug(original_img)
        
        # Визуализация комбинированных аугментаций
        show_single_augmentation(
            transforms.ToTensor()(original_img),
            combined_img,
            title="Все аугментации"
        )
        plt.savefig(f'results/{class_name}_combined.png', bbox_inches='tight')
        plt.close()

    print("Все операции завершены! Результаты сохранены в папке 'results'")

if __name__ == '__main__':
    main()
