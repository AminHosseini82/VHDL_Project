import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import os
import ast

# ==================== تنظیمات هوشمند مسیر ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ورودی: فایل CSV دیتاست
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_complete.csv')

# خروجی: پوشه تصاویر (که داخلش زیرپوشه ساخته می‌شود)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'images_heatmap')

IMG_SIZE = (224, 224)
GRID_SIZE = 32 

# ایجاد پوشه اصلی خروجی
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🚀 Starting Structured Heatmap Generation...")
print(f"📂 Reading from: {DATASET_PATH}")
print(f"📂 Saving to:   {OUTPUT_DIR}")

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded {len(df)} records.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

def create_smooth_heatmap(r_val, g_val, b_val, size=(224, 224)):
    # --- منطق تولید هیت‌مپ (همان کد قبلی) ---
    base_h, base_w = GRID_SIZE, GRID_SIZE
    r_grid = np.clip(np.random.normal(r_val, 0.15, (base_h, base_w)), 0, 1)
    g_grid = np.clip(np.random.normal(g_val, 0.12, (base_h, base_w)), 0, 1)
    b_grid = np.clip(np.random.normal(b_val, 0.08, (base_h, base_w)), 0, 1)
    
    rgb_small = np.dstack((r_grid, g_grid, b_grid)) * 255
    img_small = Image.fromarray(rgb_small.astype('uint8'), 'RGB')
    img_resized = img_small.resize(size, resample=Image.BICUBIC)
    heatmap_img = img_resized.filter(ImageFilter.GaussianBlur(radius=8))
    return heatmap_img

count = 0
for index, row in df.iterrows():
    try:
        # استخراج مقادیر (با هندل کردن فرمت‌های مختلف)
        def get_float(val):
            if isinstance(val, str) and '[' in val:
                return np.mean(ast.literal_eval(val))
            return float(val)

        # نگاشت ویژگی‌ها به کانال‌های رنگ
        ws = get_float(row.get('white_space_ratio', 0.5))
        # میانگین کنترل‌پذیری و مشاهده‌پذیری برای کانال سبز
        cc = get_float(row.get('controllability_cc0', 0.5)) 
        obs = get_float(row.get('observability_avg', 0.5))
        testability = (cc + obs) / 2
        
        cong = get_float(row.get('routing_congestion', 0.5))

        # ساخت تصویر
        img = create_smooth_heatmap(ws, testability, cong, size=IMG_SIZE)
        
        # === بخش مهم: پوشه‌بندی ===
        # دریافت لیبل (Low/Medium/High)
        label = row.get('vulnerability_label', 'Unknown')
        
        # ساخت پوشه کلاس اگر وجود نداشت
        class_dir = os.path.join(OUTPUT_DIR, label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        # ذخیره در پوشه کلاس
        # نام فایل: c17_0.png
        circuit_name = row.get('circuit_name', 'circuit')
        filename = f"{circuit_name}_{index}.png"
        save_path = os.path.join(class_dir, filename)
        
        img.save(save_path)
        
        count += 1
        if count % 500 == 0:
            print(f"   Processed {count} images...", end='\r')

    except Exception as e:
        print(f"❌ Error at index {index}: {e}")

print(f"\n✅ Done! Generated {count} images organized in '{OUTPUT_DIR}'")
