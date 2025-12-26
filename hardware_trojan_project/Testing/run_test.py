import os
import re
import sys
import numpy as np
import torch
import pickle
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from pathlib import Path

# ==================== تنظیمات پروژه ====================
CURRENT_DIR = Path.cwd()
SRC_DIR = CURRENT_DIR / "src"
COMBINED_FILE = CURRENT_DIR / "RS232_Final.v"
MODELS_DIR = CURRENT_DIR.parent / "colab_outputs" / "models"  # فرض: پوشه مدل‌ها یک مرحله عقب‌تر است

print("="*60)
print("🚀 STARTING HARDWARE TROJAN DETECTION SYSTEM")
print(f"📍 Working Directory: {CURRENT_DIR}")
print("="*60)

# ==================== مرحله ۱: ترکیب فایل‌ها ====================
def merge_files():
    print("\n[Step 1] Merging Source Files...")
    
    required_files = ["inc.h", "u_rec.v", "u_xmit.v", "uart.v"]
    
    if not SRC_DIR.exists():
        print(f"❌ Error: 'src' folder not found at {SRC_DIR}")
        sys.exit(1)

    try:
        with open(COMBINED_FILE, 'w', encoding='utf-8') as outfile:
            outfile.write("// AUTO-GENERATED FILE\n\n")
            for fname in required_files:
                fpath = SRC_DIR / fname
                if fpath.exists():
                    print(f"   📄 Adding {fname}...")
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as infile:
                        outfile.write(f"\n// === {fname} ===\n")
                        outfile.write(infile.read())
                else:
                    print(f"❌ Error: Missing file {fname}")
                    sys.exit(1)
        print(f"✅ Merged file created: {COMBINED_FILE.name}")
        return True
    except Exception as e:
        print(f"❌ Merge Failed: {e}")
        sys.exit(1)

# ==================== مرحله ۲: پردازش RTL و تولید تصویر ====================
def generate_image():
    print("\n[Step 2] Analyzing Circuit & Generating Image...")
    
    with open(COMBINED_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # استخراج ویژگی‌ها با Regex
    gates = len(re.findall(r'\b(nand|nor|xor|and|or|not|buf|always|case|if|else|reg)\b', content, re.IGNORECASE))
    wires = len(re.findall(r'\b(wire|reg|integer)\b', content, re.IGNORECASE))
    ios = len(re.findall(r'\b(input|output)\b', content, re.IGNORECASE))
    assigns = len(re.findall(r'\b(assign|<=|=)\b', content, re.IGNORECASE))
    
    print(f"   📊 Statistics: Gates={gates}, Wires={wires}, IOs={ios}")

    if gates == 0:
        print("❌ Error: Circuit seems empty!")
        sys.exit(1)

    # تبدیل به ویژگی‌های نرمالایز شده (0 تا 1)
    ws = max(0.05, 1.0 - (gates / 2000))  # White Space
    contr = min(1.0, ios / (gates + 10))  # Controllability
    observ = min(1.0, ios / (gates + 10)) # Observability
    cong = min(1.0, (wires + assigns) / (gates + 1)) # Congestion

    # تولید رنگ RGB
    r = int(ws * 220 + 20)
    g = int(((contr + observ)/2) * 220 + 20)
    b = int(cong * 220 + 20)
    
    print(f"   🎨 Generated Color (RGB): ({r}, {g}, {b})")

    # ساخت تصویر 224x224
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = r
    img_array[:, :, 1] = g
    img_array[:, :, 2] = b
    
    # اضافه کردن نویز
    noise = np.random.normal(0, 5, (224, 224, 3)).astype(int)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array, 'RGB')
    save_path = CURRENT_DIR / "circuit_image.png"
    img.save(save_path)
    print(f"✅ Image saved to: {save_path.name}")
    return img

# ==================== مرحله ۳: اجرای هوش مصنوعی ====================
def run_ai_detection(image):
    print("\n[Step 3] Running AI Detection Models...")
    
    device = torch.device('cpu')
    
    # 1. آماده‌سازی تصویر
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    try:
        # 2. لود کردن مدل CNN (ResNet18)
        print("   🧠 Loading Feature Extractor (ResNet18)...")
        cnn = models.resnet18(weights=None)
        cnn.fc = nn.Linear(cnn.fc.in_features, 3)
        
        cnn_path = MODELS_DIR / "resnet18_best.pth"
        if not cnn_path.exists():
            print(f"❌ Error: Model file missing at {cnn_path}")
            print("   Please check where your 'colab_outputs' folder is located.")
            sys.exit(1)
            
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=False))
        cnn.eval()
        
        # استخراج ویژگی
        extractor = torch.nn.Sequential(*list(cnn.children())[:-1])
        with torch.no_grad():
            features = extractor(img_tensor).view(1, -1).numpy()

        # 3. لود کردن مدل Random Forest
        print("   🌲 Loading Classifier (Random Forest)...")
        rf_path = MODELS_DIR / "ensemble_(random_forest).pkl"
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)

        # 4. پیش‌بینی
        pred = rf_model.predict(features)[0]
        probs = rf_model.predict_proba(features)[0]
        
        classes = {0: "LOW (Safe)", 1: "MEDIUM (Suspicious)", 2: "HIGH (Trojan)"}
        
        print("\n" + "="*40)
        print(f"🛑 FINAL RESULT FOR RS232-T100")
        print("="*40)
        print(f"🔍 Prediction:  {classes[pred]}")
        print(f"📊 Confidence:  {probs[pred]*100:.1f}%")
        print("-" * 20)
        print(f"   Safe:       {probs[0]*100:.1f}%")
        print(f"   Suspicious: {probs[1]*100:.1f}%")
        print(f"   Trojan:     {probs[2]*100:.1f}%")
        print("="*40)

    except Exception as e:
        print(f"❌ AI Error: {e}")

# ==================== اجرای اصلی ====================
if __name__ == "__main__":
    merge_files()
    img = generate_image()
    run_ai_detection(img)
