import os
import tkinter as tk
from tkinter import filedialog

def generate_tree(dir_path, prefix=""):
    """تابع بازگشتی برای ساختار درختی پوشه‌ها و فایل‌ها"""
    try:
        # لیست کردن محتویات پوشه و مرتب‌سازی آن‌ها
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        return ""

    tree_str = ""
    count = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = (i == count - 1)
        
        # تعیین علامت شاخه (میانی یا انتهایی)
        connector = "└── " if is_last else "├── "
        
        # اضافه کردن نام فایل یا پوشه به رشته نهایی
        tree_str += f"{prefix}{connector}{entry}\n"
        
        # اگر مورد یک پوشه بود، به صورت بازگشتی وارد آن شو
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_str += generate_tree(path, new_prefix)
            
    return tree_str

def main():
    # مخفی کردن پنجره اصلی tkinter
    root = tk.Tk()
    root.withdraw()

    # باز کردن پنجره انتخاب پوشه
    print("لطفاً پوشه مورد نظر را انتخاب کنید...")
    folder_selected = filedialog.askdirectory()

    if not folder_selected:
        print("هیچ پوشه‌ای انتخاب نشد.")
        return

    # تولید ساختار درختی
    tree_output = f"{folder_selected}\n"
    tree_output += generate_tree(folder_selected)

    # ذخیره در فایل متنی
    output_file = "folder_structure.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tree_output)

    print(f"ساختار با موفقیت در فایل {output_file} ذخیره شد.")

if __name__ == "__main__":
    main()