import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_image_data(root_dir, batch_size=64, val_split=0.1, test_split=0.2):
    """
    Load data từ folder cấu trúc: root_dir/class_name/image.png
    """
    # Chuẩn hóa theo ImageNet (Rất quan trọng cho ResNet pre-trained)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load toàn bộ dataset
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    total_size = len(full_dataset)
    classes = full_dataset.classes
    print(f"--> [INFO] Found {total_size} images. Classes: {classes}")
    
    # Tính toán kích thước split
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    # Chia tập (Random split)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"--> [INFO] Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # DataLoaders (Tận dụng num_workers vì máy bạn CPU mạnh 40 core)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader, test_loader
