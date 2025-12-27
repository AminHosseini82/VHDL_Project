import json
import os
import numpy as np
from PIL import Image, ImageFilter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'test_data')
IMG_DIR = os.path.join(CURRENT_DIR, 'test_images')
os.makedirs(IMG_DIR, exist_ok=True)

def create_heatmap():
    json_path = os.path.join(DATA_DIR, "trojan_circuit.json")
    
    if not os.path.exists(json_path):
        print("❌ Error: JSON file not found. Run step 1 first.")
        return

    print("🎨 Converting Circuit Data to Heatmap Image...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # استخراج ویژگی‌ها
    ws = data['white_space_ratio']
    cong = data['routing_congestion']
    # محاسبه قابلیت تست (Testability)
    testability = (data['observability_average'] + data['cc0_average']) / 2
    
    # تنظیمات تصویر (دقیقاً مثل مرحله آموزش)
    grid_size = 32
    img_size = (224, 224)
    
    # تولید کانال‌های رنگی (Red: WhiteSpace, Green: Testability, Blue: Congestion)
    # تروجان: Congestion بالا (Blue زیاد)، Testability کم (Green کم)، WhiteSpace کم (Red کم)
    # نتیجه باید تصویری متمایل به آبی تیره باشد
    
    r_grid = np.clip(np.random.normal(ws, 0.15, (grid_size, grid_size)), 0, 1)
    g_grid = np.clip(np.random.normal(testability, 0.12, (grid_size, grid_size)), 0, 1)
    b_grid = np.clip(np.random.normal(cong, 0.08, (grid_size, grid_size)), 0, 1)
    
    rgb = np.dstack((r_grid, g_grid, b_grid)) * 255
    img = Image.fromarray(rgb.astype('uint8'), 'RGB')
    img = img.resize(img_size, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    
    save_path = os.path.join(IMG_DIR, "trojan_heatmap.png")
    img.save(save_path)
    print(f"✅ Heatmap image saved to: {save_path}")

if __name__ == "__main__":
    create_heatmap()
