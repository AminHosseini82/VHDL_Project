import pickle
import numpy as np
import os
import sys

# تنظیم مسیرها
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_svm_model.pkl')



def load_model():
    print(f"📂 Loading pre-trained model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found. Please place 'best_svm_model.pkl' in 'models/' folder.")
        sys.exit(1)
        
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    
    # چک می‌کنیم فایل چه فرمتی دارد (بعضی وقت‌ها فقط مدل است، بعضی وقت‌ها دیکشنری)
    if isinstance(saved_data, dict) and 'model' in saved_data:
        model = saved_data['model']
        scaler = saved_data.get('scaler')
    else:
        model = saved_data
        scaler = None
        
    print("✅ Model loaded successfully!")
    return model, scaler

def predict_vulnerability(features, model, scaler=None):
    # اگر اسکیلر داشتیم، داده را نرمال می‌کنیم
    if scaler:
        features = scaler.transform(features)
        
    prediction = model.predict(features)
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    
    return prediction, probabilities

if __name__ == "__main__":
    # 1. لود مدل
    model, scaler = load_model()
    
    # 2. تولید یک داده تستی تصادفی (شبیه‌سازی یک مدار جدید)
    # فرض می‌کنیم خروجی ResNet ما 512 ویژگی دارد
    print("\n🧪 Simulating a new circuit analysis...")
    dummy_feature = np.random.rand(1, 512).astype(np.float32) 
    
    # 3. پیش‌بینی
    pred, probs = predict_vulnerability(dummy_feature, model, scaler)
    
    # 4. نمایش نتیجه
    classes = ['High', 'Low', 'Medium'] # ترتیب کلاس‌ها بر اساس آموزش شما
    predicted_class = classes[pred[0]] if isinstance(pred[0], (int, np.integer)) else pred[0]
    
    print("\n" + "="*40)
    print(f"🛑 VULNERABILITY ASSESSMENT RESULT")
    print("="*40)
    print(f"Predicted Risk Level:  [{predicted_class}]")
    
    if probs is not None:
        print(f"Confidence:            {np.max(probs)*100:.2f}%")
