import json
import os
import random

# تنظیم مسیر خروجی در پوشه rr
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'test_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_infected_circuit():
    print("⚠️  Generating Trojan-Infected Circuit Data...")
    
    # ویژگی‌های یک مدار آلوده طبق مقاله:
    # 1. تراکم بالا (Routing Congestion) -> تروجان فضا اشغال می‌کند
    # 2. مشاهده‌پذیری پایین (Observability) -> تروجان مخفی است
    # 3. فضای سفید کم (White Space) -> تروجان در فضای خالی جا سازی شده
    
    infected_features = {
        "circuit_name": "c_trojan_test_01",
        "white_space_ratio": 0.15,        # فضای خالی خیلی کم (مشکوک)
        "routing_congestion": 0.92,       # تراکم بسیار بالا (خطرناک)
        "observability_average": 0.10,    # مشاهده‌پذیری بسیار پایین (مخفی)
        "cc0_average": 0.20,              # کنترل‌پذیری پایین
        "signal_activity": 0.05,          # فعالیت سیگنال کم (تریگر تروجان معمولا خاموش است)
        "vulnerability_label": "High"     # برچسب واقعی (برای مقایسه)
    }
    
    # ذخیره در فایل JSON
    file_path = os.path.join(OUTPUT_DIR, "trojan_circuit.json")
    with open(file_path, 'w') as f:
        json.dump(infected_features, f, indent=4)
        
    print(f"✅ Infected circuit data saved to: {file_path}")
    print(f"   Features: Congestion={infected_features['routing_congestion']}, Obs={infected_features['observability_average']}")

if __name__ == "__main__":
    generate_infected_circuit()
