import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# تنظیم مسیرها برای پیدا کردن مدل
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) # رفتن به ریشه پروژه
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_svm_model.pkl')
IMG_PATH = os.path.join(CURRENT_DIR, 'test_images', 'trojan_heatmap.png')

def run_diagnosis():
    print("\n🔍 STARTING HARDWARE TROJAN DIAGNOSIS...")
    
    # 1. بررسی فایل‌ها
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(IMG_PATH):
        print(f"❌ Error: Image not found at {IMG_PATH}. Run step 2 first.")
        return

    # 2. لود مدل ResNet برای استخراج ویژگی
    print("   Initializing Feature Extractor (ResNet18)...")
    device = torch.device("cpu") # برای تست تکی CPU کافی است
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()

    # 3. پردازش تصویر
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(IMG_PATH).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) # افزودن بعد batch

    # 4. استخراج ویژگی
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.view(features.size(0), -1).numpy()
    
    print("   Features extracted successfully.")

    # 5. لود مدل SVM و پیش‌بینی
    print(f"   Loading AI Model from: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        
    if isinstance(saved_data, dict):
        model = saved_data['model']
        scaler = saved_data.get('scaler')
    else:
        model = saved_data
        scaler = None

    # نرمال‌سازی ویژگی
    if scaler:
        features = scaler.transform(features)

    # پیش‌بینی
    prediction = model.predict(features)
    
    # نمایش نتیجه
    classes = ['High', 'Low', 'Medium'] # ترتیب استاندارد
    result = classes[prediction[0]]
    
    print("\n" + "="*50)
    print(f"🛑 DIAGNOSIS RESULT for 'c_trojan_test_01'")
    print("="*50)
    
    if result == "High":
        print(f"⚠️  ALERT: HIGH VULNERABILITY DETECTED! ")
        print("   The system identified potential Trojan characteristics.")
        print("   (High Congestion + Low Observability)")
    elif result == "Medium":
        print(f"⚠️  WARNING: Medium Vulnerability.")
    else:
        print(f"✅  SAFE: Low Vulnerability.")
        
    print("="*50)

if __name__ == "__main__":
    run_diagnosis()
