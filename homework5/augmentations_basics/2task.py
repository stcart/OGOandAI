import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms

# НОВЫЕ КАСТОМНЫЕ АУГМЕНТАЦИИ
class RandomRaindrops:
    """Имитация капель дождя на изображении"""
    def __init__(self, p=0.5, drop_count=20, drop_length=15, drop_width=2):
        self.p = p
        self.drop_count = drop_count
        self.drop_length = drop_length
        self.drop_width = drop_width

    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        for _ in range(self.drop_count):
            x = random.randint(0, w)
            y = random.randint(0, h)
            length = random.randint(5, self.drop_length)
            
            # Рисуем "каплю" как белый отрезок с размытием
            cv2.line(img_np, (x, y), (x, y+length), 
                    (255, 255, 255), self.drop_width)
            
        return Image.fromarray(cv2.GaussianBlur(img_np, (3, 3), 0))

class ColorShift:
    """Случайный сдвиг цветовых каналов"""
    def __init__(self, p=0.5, max_shift=10):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Сдвигаем каждый канал на случайное значение
        for i in range(3):
            dx = random.randint(-self.max_shift, self.max_shift)
            dy = random.randint(-self.max_shift, self.max_shift)
            
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img_np[:,:,i] = cv2.warpAffine(img_np[:,:,i], M, (w, h))
            
        return Image.fromarray(img_np)

class VintageEffect:
    """Винтажный эффект (сепия + шум + виньетирование)"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Эффект сепии
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        img_np = cv2.transform(img_np, sepia_kernel)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Виньетирование
        X, Y = np.ogrid[:h, :w]
        vignette = 1 - np.sqrt((X - h/2)**2 + (Y - w/2)**2) / np.sqrt((h/2)**2 + (w/2)**2)
        vignette = np.clip(vignette * 0.6 + 0.4, 0, 1)
        for i in range(3):
            img_np[:,:,i] = img_np[:,:,i] * vignette
            
        return Image.fromarray(img_np)

# СРАВНЕНИЕ С АУГМЕНТАЦИЯМИ ИЗ EXTRA_AUGS.PY
def compare_augmentations(dataset, num_samples=3):
    # Выбираем случайные изображения
    samples = [dataset[i] for i in random.sample(range(len(dataset)), num_samples)]
    
    # Готовые аугментации из extra_augs.py
    extra_augs = [
        ("GaussianNoise", transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.2)
        ])),
        ("RandomErasing", transforms.Compose([
            transforms.ToTensor(),
            RandomErasingCustom(p=1.0)
        ])),
        ("CutOut", transforms.Compose([
            transforms.ToTensor(),
            CutOut(p=1.0, size=(32, 32))
        ]))
    ]
    
    # Наши новые аугментации
    custom_augs = [
        ("Raindrops", RandomRaindrops(p=1.0)),
        ("ColorShift", ColorShift(p=1.0)),
        ("Vintage", VintageEffect(p=1.0))
    ]
    
    # Применяем все аугментации
    for img, label in samples:
        plt.figure(figsize=(18, 6))
        plt.suptitle(f"Class: {dataset.get_class_names()[label]}", fontsize=16)
        
        # Оригинал
        plt.subplot(1, 7, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis('off')
        
        # Готовые аугментации
        for i, (name, aug) in enumerate(extra_augs, start=2):
            aug_img = aug(img).permute(1, 2, 0).numpy()
            plt.subplot(1, 7, i)
            plt.imshow(np.clip(aug_img, 0, 1))
            plt.title(f"Built-in:\n{name}")
            plt.axis('off')
        
        # Наши аугментации
        for i, (name, aug) in enumerate(custom_augs, start=5):
            aug_img = aug(img)
            plt.subplot(1, 7, i)
            plt.imshow(aug_img)
            plt.title(f"Custom:\n{name}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# ЗАПУСК СРАВНЕНИЯ
if __name__ == "__main__":
    # Загружаем датасет
    dataset = CustomImageDataset(TRAIN_PATH, transform=None, target_size=(224, 224))
    
    # Сравниваем аугментации
    print("Сравнение кастомных и встроенных аугментаций:")
    compare_augmentations(dataset)
