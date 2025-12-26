import numpy as np
from PIL import Image, ImageFilter
import os

# ==================== تنظیمات تست ====================
OUTPUT_TEST_FILE = "test_heatmap_sample.png"
IMG_SIZE = (224, 224)
GRID_SIZE = 32  # سایز گرید پایه

print("🧪 Running Heatmap Generation Test...")

def create_smooth_heatmap_test(r_val, g_val, b_val, size=(224, 224)):
    """
    نسخه تستی تابع تولید هیت‌مپ
    """
    print(f"   Inputs -> R (Space): {r_val}, G (Testability): {g_val}, B (Congestion): {b_val}")
    
    # 1. ساخت ماتریس پایه (Base Grid)
    # مقادیر را کمی حول عدد اصلی تغییر می‌دهیم تا بافت (Texture) ایجاد شود
    base_h, base_w = GRID_SIZE, GRID_SIZE
    
    # کانال قرمز: واریانس بیشتر (ابر و باد بیشتر)
    r_grid = np.random.normal(r_val, 0.15, (base_h, base_w))
    
    # کانال سبز: واریانس متوسط
    g_grid = np.random.normal(g_val, 0.12, (base_h, base_w))
    
    # کانال آبی: واریانس کمتر
    b_grid = np.random.normal(b_val, 0.08, (base_h, base_w))
    
    # محدود کردن بین 0 و 1
    r_grid = np.clip(r_grid, 0, 1)
    g_grid = np.clip(g_grid, 0, 1)
    b_grid = np.clip(b_grid, 0, 1)
    
    # 2. تبدیل به RGB اولیه
    rgb_small = np.dstack((r_grid, g_grid, b_grid)) * 255
    img_small = Image.fromarray(rgb_small.astype('uint8'), 'RGB')
    
    # 3. بزرگ‌نمایی نرم (Bicubic)
    img_resized = img_small.resize(size, resample=Image.BICUBIC)
    
    # 4. اعمال بلور نهایی (Gaussian Blur)
    # شعاع 8 تا 10 برای سایز 224 عالی است
    heatmap_img = img_resized.filter(ImageFilter.GaussianBlur(radius=8))
    
    return heatmap_img

# اجرای تست با مقادیر فرضی (مثلاً یک مدار با شلوغی متوسط و فضای خالی کم)
# R=0.3 (فضای خالی کم), G=0.6 (تست‌پذیری خوب), B=0.8 (شلوغی زیاد - آبی تند)
test_img = create_smooth_heatmap_test(0.3, 0.6, 0.8, size=IMG_SIZE)

# ذخیره تصویر
test_img.save(OUTPUT_TEST_FILE)
print(f"✅ Test Image Saved: {os.path.abspath(OUTPUT_TEST_FILE)}")
print("   Please open this image and check if it looks like a smooth heatmap.")
