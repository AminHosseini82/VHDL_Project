import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path # این خط مهم است

# ==================== تعریف مسیرهای اصلی ====================
# این بخش اضافه شده است
PROJECT_ROOT = Path(r"F:\Amin_Projects\University\VHDL\hardware_trojan_project")
# ==========================================================

print("="*80)
print("TEST 3: CREATE TEST CIRCUIT WITH TROJAN (c17 + MERS)")
print("="*80 + "\n")

# مشخصات مدار c17 با تروجال MERS
test_circuit_info = {
    "circuit_name": "c17",
    "trojan_type": "MERS (Multiplexed Externally controlled Reroute Switch)",
    "trojan_location": "Critical signal path",
    "vulnerability_score": 0.85,  # امتیاز تروجال (0-1)
    "expected_class": "High",      # انتظار: کلاس High
    "features": {
        "gates": 12,
        "nets": 18,
        "inputs": 5,
        "outputs": 2,
        "white_space": 0.35,
        "controllability": 0.72,
        "observability": 0.68,
        "signal_probability": 0.55,
        "routing_congestion": 0.78,
    }
}

print("📋 Test Circuit Information:")
print(f"   Circuit Name: {test_circuit_info['circuit_name']}")
print(f"   Trojan Type: {test_circuit_info['trojan_type']}")
print(f"   Trojan Location: {test_circuit_info['trojan_location']}")
print(f"   Vulnerability Score: {test_circuit_info['vulnerability_score']:.2f} (0-1)")
print(f"   Expected Class: {test_circuit_info['expected_class']}")

print("\n🔧 Circuit Features:")
for key, value in test_circuit_info['features'].items():
    print(f"   {key.ljust(25)}: {value}")

# ایجاد تصویر RGB مصنوعی برای این مدار
print("\n🖼️  Generating RGB Image (224x224)...")

# ایجاد ماتریس‌های RGB بر اساس ویژگی‌های مدار
features = test_circuit_info['features']

# R channel: White Space + Testability
r_channel = np.ones((224, 224), dtype=np.uint8) * int(features['white_space'] * 200 + 30)

# G channel: Controllability + Observability + Signal Activity
g_value = (features['controllability'] + features['observability'] + features['signal_probability']) / 3
g_channel = np.ones((224, 224), dtype=np.uint8) * int(g_value * 200 + 30)

# B channel: Routing Congestion
b_channel = np.ones((224, 224), dtype=np.uint8) * int(features['routing_congestion'] * 200 + 30)

# اضافه کردن نویز و الگو برای واقع‌گرایی
noise = np.random.normal(0, 10, (224, 224))
r_channel = np.clip(r_channel + noise, 0, 255).astype(np.uint8)
g_channel = np.clip(g_channel + noise, 0, 255).astype(np.uint8)
b_channel = np.clip(b_channel + noise, 0, 255).astype(np.uint8)

# ترکیب کانال‌ها
test_image = np.stack([r_channel, g_channel, b_channel], axis=2)

print(f"   ✅ Image created: {test_image.shape}")
print(f"   R channel: mean={r_channel.mean():.1f}, std={r_channel.std():.1f}")
print(f"   G channel: mean={g_channel.mean():.1f}, std={g_channel.std():.1f}")
print(f"   B channel: mean={b_channel.mean():.1f}, std={b_channel.std():.1f}")

# ذخیره تصویر تست
# حالا این خط بدون ارور اجرا می‌شود
test_image_path = PROJECT_ROOT / "test_image_c17_trojan.png"
test_image_pil = Image.fromarray(test_image, mode='RGB')
test_image_pil.save(test_image_path)

print(f"\n✅ Test image saved: {test_image_path}")
