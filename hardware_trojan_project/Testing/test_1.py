import os
from pathlib import Path

# تعریف مسیرهای اصلی
PROJECT_ROOT = Path(r"F:\Amin_Projects\University\VHDL\hardware_trojan_project")
COLAB_OUTPUTS = PROJECT_ROOT / "colab_outputs"

print("="*80)
print("TEST 1: CHECKING FILES EXISTENCE")
print("="*80 + "\n")

# لیست فایل‌هایی که باید موجود باشند
required_files = {
    "Models": [
        "models/resnet18_best.pth",
        "models/ensemble_(random_forest).pkl",
        "models/ensemble_(gradient_boosting).pkl",
        "models/svm.pkl",
        "models/knn_(k=5).pkl",
        "models/naive_bayes.pkl",
    ],
    "Results": [
        "final_report/01_all_comparisons.png",
        "final_report/02_improvement_analysis.png",
        "final_report/comparison_table.csv",
        "final_report/final_report.txt",
    ],
    "Dataset": [
        "dataset/dataset_complete.csv",
        "dataset/dataset_train.csv",
        "dataset/dataset_val.csv",
        "dataset/dataset_test.csv",
    ]
}

# بررسی فایل‌ها
all_exists = True
for category, files in required_files.items():
    print(f"📁 {category}:")
    for file_path in files:
        full_path = COLAB_OUTPUTS / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024*1024)
            print(f"  ✅ {file_path.ljust(45)} ({size_mb:6.2f} MB)")
        else:
            print(f"  ❌ {file_path.ljust(45)} MISSING!")
            all_exists = False
    print()

if all_exists:
    print("✅ ALL FILES FOUND!")
else:
    print("❌ SOME FILES ARE MISSING!")
