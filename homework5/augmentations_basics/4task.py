class AugmentationPipeline:
    """Исправленный пайплайн с обработкой PIL.Image и torch.Tensor"""
    def __init__(self):
        self.augmentations = {}
        
    def add_augmentation(self, name, augmentation):
        self.augmentations[name] = augmentation
        return self
    
    def remove_augmentation(self, name):
        if name in self.augmentations:
            del self.augmentations[name]
        return self
    
    def apply(self, image):
        # Конвертируем в PIL.Image если нужно
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Применяем аугментации
        for name, aug in self.augmentations.items():
            try:
                if isinstance(aug, (transforms.transforms.Transform, torch.nn.Module)):
                    # Для torchvision transforms
                    image = aug(image)
                else:
                    # Для кастомных функций
                    image = aug(image)
            except Exception as e:
                print(f"Ошибка в аугментации {name}: {e}")
                continue
                
        return image
    
    def get_augmentations(self):
        return self.augmentations.copy()

# Обновленные конфигурации с исправлениями
def create_light_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.3))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1))
    return pipeline

def create_medium_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    pipeline.add_augmentation("random_rotate", transforms.RandomRotation(15))
    pipeline.add_augmentation("random_blur", transforms.GaussianBlur(3))
    return pipeline

def create_heavy_pipeline():
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation("random_flip", transforms.RandomHorizontalFlip(p=0.7))
    pipeline.add_augmentation("color_jitter", transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    pipeline.add_augmentation("random_rotate", transforms.RandomRotation(30))
    pipeline.add_augmentation("random_blur", transforms.GaussianBlur(5))
    return pipeline

# Обновленная функция визуализации
def apply_and_save_pipelines(dataset, pipelines, num_samples=3):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    samples = [dataset[i] for i in range(num_samples)]
    
    for pipe_name, pipeline in pipelines.items():
        print(f"\nApplying {pipe_name} pipeline...")
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i, (image, label) in enumerate(samples):
            # Оригинал
            plt.subplot(num_samples, 2, 2*i+1)
            plt.imshow(image)
            plt.title(f"Original (class: {dataset.get_class_names()[label]})")
            plt.axis('off')
            
            # Аугментированное
            try:
                aug_img = pipeline.apply(image)
                if not isinstance(aug_img, Image.Image):
                    aug_img = transforms.ToPILImage()(aug_img)
                
                plt.subplot(num_samples, 2, 2*i+2)
                plt.imshow(aug_img)
                plt.title(f"Augmented ({pipe_name})")
                plt.axis('off')
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_PATH, f"{pipe_name}_augmentations.jpg")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

# Запуск
if __name__ == "__main__":
    dataset = CustomImageDataset(TRAIN_PATH, transform=None, target_size=(224, 224))
    
    pipelines = {
        "light": create_light_pipeline(),
        "medium": create_medium_pipeline(), 
        "heavy": create_heavy_pipeline()
    }
    
    apply_and_save_pipelines(dataset, pipelines)
    print("\nDone! Results saved in", RESULTS_PATH)
