import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from tqdm import tqdm
import multiprocessing

# ==================== تنظیمات ====================
BATCH_SIZE = 16  # کمتر کردم تا رم پر نشود
IMG_SIZE = 224

def main():
    # پیدا کردن مسیرها
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'images_heatmap')
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'features.pkl')

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("🚀 Starting Feature Extraction (Safe Mode)...")
    print(f"📂 Reading images from: {DATA_DIR}")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("❌ Error: Image directory is empty or missing!")
        exit()

    # تنظیم دستگاه (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    # لود مدل ResNet18
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # آماده‌سازی دیتا
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        # نکته کلیدی: در ویندوز بهتر است num_workers=0 باشد تا کرش نکند
        # اگر سیستم قوی دارید، می‌توانید num_workers=2 بگذارید ولی حتما داخل main
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"✅ Found {len(dataset)} images in classes: {dataset.classes}")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        exit()

    # استخراج ویژگی
    all_features = []
    all_labels = []

    print("⏳ Extracting features...")

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1).cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())

    # ذخیره‌سازی
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    print(f"📊 Feature Matrix Shape: {X.shape}")
    
    data_to_save = {
        'features': X,
        'labels': y,
        'class_names': dataset.classes
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"✅ Success! Features saved to: {OUTPUT_FILE}")

# محافظت برای ویندوز
if __name__ == '__main__':
    multiprocessing.freeze_support() # اختیاری برای امنیت بیشتر
    main()
