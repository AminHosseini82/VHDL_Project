import torch
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"F:\Amin_Projects\University\VHDL\hardware_trojan_project")
MODELS_DIR = PROJECT_ROOT / "colab_outputs" / "models"

print("="*80)
print("TEST 2: LOADING MODELS")
print("="*80 + "\n")

try:
    # بارگذاری CNN
    print("1️⃣  Loading CNN Model (ResNet-18)...")
    cnn_model = torch.load(MODELS_DIR / "resnet18_best.pth", map_location='cpu')
    print("   ✅ CNN Model loaded successfully!")
    
    # بارگذاری Random Forest
    print("\n2️⃣  Loading Random Forest...")
    with open(MODELS_DIR / "ensemble_(random_forest).pkl", 'rb') as f:
        rf_model = pickle.load(f)
    print("   ✅ Random Forest loaded successfully!")
    print(f"   Number of trees: {rf_model.n_estimators}")
    
    # بارگذاری Gradient Boosting
    print("\n3️⃣  Loading Gradient Boosting...")
    with open(MODELS_DIR / "ensemble_(gradient_boosting).pkl", 'rb') as f:
        gb_model = pickle.load(f)
    print("   ✅ Gradient Boosting loaded successfully!")
    
    # بارگذاری SVM
    print("\n4️⃣  Loading SVM...")
    with open(MODELS_DIR / "svm.pkl", 'rb') as f:
        svm_model = pickle.load(f)
    print("   ✅ SVM loaded successfully!")
    
    # بارگذاری KNN
    print("\n5️⃣  Loading KNN...")
    with open(MODELS_DIR / "knn_(k=5).pkl", 'rb') as f:
        knn_model = pickle.load(f)
    print("   ✅ KNN loaded successfully!")
    
    # بارگذاری Naive Bayes
    print("\n6️⃣  Loading Naive Bayes...")
    with open(MODELS_DIR / "naive_bayes.pkl", 'rb') as f:
        nb_model = pickle.load(f)
    print("   ✅ Naive Bayes loaded successfully!")
    
    print("\n" + "="*80)
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("="*80)

except Exception as e:
    print(f"❌ ERROR: {str(e)}")
