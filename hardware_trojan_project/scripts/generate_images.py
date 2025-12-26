import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import os
import ast
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# ==================== تنظیمات ====================
DATASET_PATH = "../data/dataset.csv"
OUTPUT_DIR = "../data/images"
IMG_SIZE = (224, 224)
GRID_SIZE = 32  # سایز گرید اولیه (مثلا 32x32) که بعدا به 224x224 تبدیل و مات می‌شود

# ایجاد پوشه خروجی اگر وجود نداشته باشد
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🚀 Starting Heatmap Image Generation...")

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"📂 Loaded dataset with {len(df)} records.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

def create_smooth_heatmap(r_val, g_val, b_val, size=(224, 224)):
    """
    این تابع به جای پیکسل‌های ساده، یک تصویر هیت‌مپ نرم تولید می‌کند.
    ما ابتدا یک ماتریس تصادفی کوچک می‌سازیم و سپس آن را بزرگ و بلور می‌کنیم
    تا شبیه نقشه حرارتی واقعی شود.
    """
    
    # 1. ساخت ماتریس پایه (Base Grid) با ابعاد کوچک
    # برای اینکه پترن‌های رندوم داشته باشیم، مقادیر را کمی حول میانگین تغییر می‌دهیم
    base_w = GRID_SIZE
    base_h = GRID_SIZE
    
    # تولید نویز نرمال حول مقدار اصلی برای هر کانال
    # r_val, g_val, b_val اعداد بین 0 تا 1 هستند
    
    # کانال قرمز (White Space) - شدت تغییرات بیشتر
    r_grid = np.random.normal(r_val, 0.15, (base_h, base_w))
    
    # کانال سبز (Controllability) - شدت تغییرات متوسط
    g_grid = np.random.normal(g_val, 0.1, (base_h, base_w))
    
    # کانال آبی (Congestion) - شدت تغییرات کمتر (معمولا یکنواخت‌تر)
    b_grid = np.random.normal(b_val, 0.05, (base_h, base_w))
    
    # محدود کردن مقادیر بین 0 و 1
    r_grid = np.clip(r_grid, 0, 1)
    g_grid = np.clip(g_grid, 0, 1)
    b_grid = np.clip(b_grid, 0, 1)
    
    # 2. تبدیل به تصویر RGB اولیه (کوچک)
    rgb_small = np.dstack((r_grid, g_grid, b_grid)) * 255
    img_small = Image.fromarray(rgb_small.astype('uint8'), 'RGB')
    
    # 3. بزرگ‌نمایی با اینترپولیشن (BICUBIC) برای نرم شدن اولیه
    img_resized = img_small.resize(size, resample=Image.BICUBIC)
    
    # 4. اعمال فیلتر گاشن (Gaussian Blur) برای ایجاد حالت هیت‌مپ کامل
    # شعاع بلور (Radius) تعیین می‌کند چقدر تصویر نرم شود
    heatmap_img = img_resized.filter(ImageFilter.GaussianBlur(radius=8))
    
    return heatmap_img

count = 0
for index, row in df.iterrows():
    try:
        # دریافت مقادیر ویژگی‌ها از دیتاست
        # فرض بر این است که ستون‌های features نرمالایز شده هستند (0 تا 1)
        # اگر ستون‌ها لیست هستند، میانگین آن‌ها را می‌گیریم
        
        # مدیریت فرمت‌های مختلف ذخیره شده در CSV
        def get_val(col_name):
            val = row[col_name]
            if isinstance(val, str):
                if '[' in val: # اگر لیست است
                    val_list = ast.literal_eval(val)
                    return np.mean(val_list)
                else:
                    return float(val)
            return float(val)

        # استخراج ویژگی‌ها (تطبیق با نام ستون‌های فایل CSV شما)
        # اگر نام ستون‌ها فرق دارد، اینجا را تغییر دهید
        if 'white_space' in row:
            ws = get_val('white_space')
        else:
            ws = 0.5 # مقدار پیش‌فرض
            
        if 'controllability' in row:
            cont = get_val('controllability')
            obs = get_val('observability') if 'observability' in row else 0.5
            testability = (cont + obs) / 2
        else:
            testability = 0.5

        if 'routing_congestion' in row:
            cong = get_val('routing_congestion')
        else:
            cong = 0.5
            
        # تولید تصویر هیت‌مپ
        img = create_smooth_heatmap(r_val=ws, g_val=testability, b_val=cong, size=IMG_SIZE)
        
        # نام‌گذاری فایل: benchmark_index_label.png
        bench_name = row['benchmark'] if 'benchmark' in row else 'unknown'
        label = row['trojan_label'] if 'trojan_label' in row else 'unknown'
        
        filename = f"{bench_name}_{index}_{label}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        img.save(save_path)
        
        count += 1
        if count % 100 == 0:
            print(f"   Processed {count} images...", end='\r')
            
    except Exception as e:
        print(f"❌ Error processing row {index}: {e}")
        continue

print(f"\n✅ Done! Generated {count} heatmap images in '{OUTPUT_DIR}'")
