import os
import sys
import pandas as pd


INPUT_DIR = "data/works_lpdr/.beforeMerged"
OUTPUT_DIR = "data/works_lpdr/.afterMerged"
skiprows = 0

# 读取所有xlsx文件保存为csv
for file in os.listdir(INPUT_DIR):
    if file.endswith(".xlsx"):
        file_path = os.path.join(INPUT_DIR, file)
        df = pd.read_excel(file_path, skiprows=skiprows)
        csv_file_path = os.path.join(OUTPUT_DIR, file.replace(".xlsx", ".csv"))
        df.to_csv(csv_file_path, index=False)
        print(f"已将 {file} 转为 {csv_file_path}")
        # 删除xlsx
        os.remove(file_path)
