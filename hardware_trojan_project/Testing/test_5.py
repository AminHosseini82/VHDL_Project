import pandas as pd
import numpy as np
import torch
import pickle
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from pathlib import Path

# ==================== تنظیمات و مسیرها ====================
PROJECT_ROOT = Path(r"F:\Amin_Projects\University\VHDL\hardware_trojan_project")
MODELS_DIR = PROJECT_ROOT / "colab_outputs" / "models"
FEATURES_PATH = PROJECT_ROOT / "test_features.npy"
TEST_IMAGE_PATH = PROJECT_ROOT / "test_image_c17_trojan.png"

device = torch.device('cpu')
# ========================================================

print("="*80)
print("TEST 5: PREDICTION WITH ALL MODELS")
print("="*80 + "\n")

# 1. بارگذاری ویژگی‌ها (Test Features)
if not FEATURES_PATH.exists():
    print("❌ Error: 'test_features.npy' not found! Run test_4.py first.")
    exit()

test_features = np.load(FEATURES_PATH)
print(f"✅ Features loaded: {test_features.shape}")

# 2. بارگذاری مدل‌ها (Models Loading)
print("\n🔄 Loading Models...")

# CNN (ResNet18)
try:
    cnn_model = models.resnet18(weights=None)
    num_ftrs = cnn_model.fc.in_features
    cnn_model.fc = nn.Linear(num_ftrs, 3)
    state_dict = torch.load(MODELS_DIR / "resnet18_best.pth", map_location=device, weights_only=False)
    cnn_model.load_state_dict(state_dict)
    cnn_model.to(device)
    cnn_model.eval()
    print("   ✅ CNN loaded")
except Exception as e:
    print(f"   ❌ Error loading CNN: {e}")
    exit()

# Other Classifiers
classifiers = {}
for name, file in [
    ("Random Forest", "ensemble_(random_forest).pkl"),
    ("Gradient Boosting", "ensemble_(gradient_boosting).pkl"),
    ("SVM", "svm.pkl"),
    ("KNN", "knn_(k=5).pkl"),
    ("Naive Bayes", "naive_bayes.pkl")
]:
    try:
        with open(MODELS_DIR / file, 'rb') as f:
            classifiers[name] = pickle.load(f)
        print(f"   ✅ {name} loaded")
    except Exception as e:
        print(f"   ❌ Error loading {name}: {e}")

# ==================== شروع پیش‌بینی ====================

class_names = {0: "Low", 1: "Medium", 2: "High"}
expected_class = "High"
predictions = {}

print("\n🤖 Running Predictions...\n")

# 1. CNN Prediction
print("1️⃣  CNN (ResNet-18):")
# برای پیش‌بینی با CNN، نیاز به خود تصویر داریم، نه فقط ویژگی‌ها
# اما چون در test_4 ویژگی استخراج کردیم، می‌توانیم از همان ویژگی‌ها برای دیگر مدل‌ها استفاده کنیم
# برای CNN باید دوباره تصویر را پردازش کنیم یا خروجی softmax را بگیریم.
# اینجا برای سادگی، تصویر را دوباره لود می‌کنیم تا خروجی دقیق بگیریم:

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img = Image.open(TEST_IMAGE_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    cnn_output = cnn_model(img_tensor)
    cnn_probs = torch.softmax(cnn_output, dim=1).cpu().numpy()[0]
    cnn_pred_class = np.argmax(cnn_probs)
    cnn_confidence = cnn_probs[cnn_pred_class] * 100

predictions['CNN'] = {
    'class': class_names[cnn_pred_class],
    'confidence': cnn_confidence,
    'probabilities': {k: v*100 for k, v in zip(['Low', 'Medium', 'High'], cnn_probs)}
}
print(f"   Predicted: {class_names[cnn_pred_class]} ({cnn_confidence:.2f}%)")

# 2. Other Classifiers Prediction
i = 2
for name, model in classifiers.items():
    print(f"\n{i}️⃣  {name}:")
    
    # Predict Class
    pred_class_idx = model.predict(test_features)[0]
    
    # Predict Probabilities (if supported)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test_features)[0]
        confidence = probs[pred_class_idx] * 100
    elif name == "SVM": # SVM needs special handling for prob
         # For SVM without probability=True, we use decision_function
         d_func = model.decision_function(test_features)[0]
         probs = np.exp(d_func) / np.sum(np.exp(d_func)) # Softmax approximation
         confidence = probs[int(pred_class_idx)] * 100
    else:
        probs = [0, 0, 0] # Fallback
        confidence = 100.0

    predictions[name] = {
        'class': class_names[int(pred_class_idx)],
        'confidence': confidence,
        'probabilities': {k: v*100 for k, v in zip(['Low', 'Medium', 'High'], probs)}
    }
    
    print(f"   Predicted: {class_names[int(pred_class_idx)]} ({confidence:.2f}%)")
    i += 1

# ==================== گزارش نهایی ====================
print("\n" + "="*80)
print("📊 FINAL RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame({
    'Model': list(predictions.keys()),
    'Predicted': [p['class'] for p in predictions.values()],
    'Confidence': [f"{p['confidence']:.2f}%" for p in predictions.values()],
    'Result': ["✅ CORRECT" if p['class'] == expected_class else "❌ WRONG" for p in predictions.values()]
})

print(results_df.to_string(index=False))
print("\n✅ DONE!")
