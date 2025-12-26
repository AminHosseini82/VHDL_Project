import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn  # اضافه شده برای تغییر لایه آخر
from PIL import Image
from pathlib import Path
import numpy as np

# ==================== تعریف مسیرها ====================
PROJECT_ROOT = Path(r"F:\Amin_Projects\University\VHDL\hardware_trojan_project")
MODELS_DIR = PROJECT_ROOT / "colab_outputs" / "models"
test_image_path = PROJECT_ROOT / "test_image_c17_trojan.png"
# =====================================================

print("="*80)
print("TEST 4: EXTRACT CNN FEATURES")
print("="*80 + "\n")

# 1. ساخت معماری مدل ResNet18
print("1️⃣  Building ResNet-18 Architecture...")
try:
    # دانلود نسخه خام ResNet18 (بدون وزن اولیه، چون وزن‌های خودمان را داریم)
    cnn_model = models.resnet18(weights=None)
    
    # تغییر لایه آخر (fc) برای تطابق با مدل آموزش‌دیده (3 کلاس: Low, Medium, High)
    num_ftrs = cnn_model.fc.in_features
    cnn_model.fc = nn.Linear(num_ftrs, 3)
    
    print("   ✅ Architecture created!")
    
    # 2. بارگذاری وزن‌های ذخیره شده
    print("2️⃣  Loading Trained Weights...")
    weights_path = MODELS_DIR / "resnet18_best.pth"
    
    # بارگذاری دیکشنری وزن‌ها
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # بارگذاری وزن‌ها روی مدل
    cnn_model.load_state_dict(state_dict)
    print("   ✅ Weights loaded successfully!")

except Exception as e:
    print(f"   ❌ Error: {e}")
    exit()

# 3. تنظیم مدل برای اجرا
device = torch.device('cpu')
cnn_model = cnn_model.to(device)
cnn_model.eval()

# 4. حذف لایه آخر برای استخراج ویژگی‌ها (Feature Extraction)
# ما لایه fc (آخرین لایه) را حذف می‌کنیم تا بردار ویژگی 512 تایی بگیریم
feature_extractor = torch.nn.Sequential(*list(cnn_model.children())[:-1])

# 5. پیش‌پردازش (Preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 6. بارگذاری تصویر تست
if not test_image_path.exists():
    print(f"❌ Error: Test image not found at {test_image_path}")
    exit()

test_image = Image.open(test_image_path).convert('RGB')
test_image_tensor = transform(test_image).unsqueeze(0).to(device)

print("\n🔧 Extracting 512-D feature vector...")

# 7. استخراج ویژگی
with torch.no_grad():
    features = feature_extractor(test_image_tensor)
    features = features.view(features.size(0), -1) # Flatten (1, 512, 1, 1) -> (1, 512)
    features_np = features.cpu().numpy()

print(f"   ✅ Features extracted!")
print(f"   Shape: {features_np.shape}")
print(f"   First 10 features: {features_np[0, :10]}")
print(f"   Mean: {features_np.mean():.4f}")
print(f"   Std: {features_np.std():.4f}")

# 8. ذخیره ویژگی‌ها برای مرحله بعد
np.save(PROJECT_ROOT / "test_features.npy", features_np)
print(f"\n✅ Features saved to: {PROJECT_ROOT / 'test_features.npy'}")
