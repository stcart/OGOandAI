import os
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Конфигурация
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Проверка и создание папок
def check_data_paths():
    # Пути к данным (проверяем оба варианта)
    possible_paths = [
        ('data/train', 'data/val'),
        ('/content/data/train', '/content/data/val')
    ]
    
    for train_path, val_path in possible_paths:
        if os.path.exists(train_path) and os.path.exists(val_path):
            return train_path, val_path
    
    # Если val не найден, создаем его из train
    for train_path, _ in possible_paths:
        if os.path.exists(train_path):
            print(f"Папка val не найдена, создаем split из train...")
            return train_path, create_val_split(train_path)
    
    raise FileNotFoundError("Не найдены папки с данными train/val")

def create_val_split(train_path, val_ratio=0.2):
    """Создает val из train если папка val отсутствует"""
    val_path = train_path.replace('train', 'val')
    os.makedirs(val_path, exist_ok=True)
    
    for class_name in os.listdir(train_path):
        class_train_path = os.path.join(train_path, class_name)
        class_val_path = os.path.join(val_path, class_name)
        
        if not os.path.isdir(class_train_path):
            continue
            
        os.makedirs(class_val_path, exist_ok=True)
        images = os.listdir(class_train_path)
        val_samples = max(1, int(len(images) * val_ratio))  # Хотя бы 1 изображение
        
        for img in random.sample(images, val_samples):
            src = os.path.join(class_train_path, img)
            dst = os.path.join(class_val_path, img)
            shutil.copy2(src, dst)  # Копируем вместо перемещения
    
    return val_path

# Подготовка данных
def prepare_data():
    train_path, val_path = check_data_paths()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomImageDataset(train_path, transform=transform)
    val_dataset = CustomImageDataset(val_path, transform=transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, train_dataset.get_class_names()

# Подготовка модели
def prepare_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Замораживаем все слои кроме последнего
    for param in model.parameters():
        param.requires_grad = False
        
    # Заменяем последний слой
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.requires_grad = True
    
    return model.to(DEVICE)

# Обучение и валидация
def train_model(model, train_loader, val_loader, class_names):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS} | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}%')
    
    return history

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

# Визуализация
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'training_metrics.jpg'))
    plt.show()

# Главная функция
def main():
    try:
        train_loader, val_loader, class_names = prepare_data()
        print(f"Classes: {class_names}")
        
        model = prepare_model(len(class_names))
        print(model)
        
        history = train_model(model, train_loader, val_loader, class_names)
        plot_history(history)
        
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Проверьте структуру папок. Ожидается:")
        print("data/train/class1/...")
        print("data/val/class1/...")

if __name__ == "__main__":
    import shutil
    import random
    main()
