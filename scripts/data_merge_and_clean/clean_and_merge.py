import os
import sys
import pandas as pd
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.dataProcessing import data_cleaning

#####################读取.berforeMerged目录下所有.csv和.xlsx文件，清洗后合并到.AfterMerged目录下
skiprows = 0
#####################把要合并的文件放在.beforeMerged目录下

if __name__ == "__main__":
    INPUT_DIR = "data/works_lpdr/.beforeMerged"
    OUTPUT_DIR = "data/works_lpdr/.afterMerged"
    merged_output = os.path.join(OUTPUT_DIR, "merged.csv")

    # 获取输入目录下所有 .csv 和 .xlsx 文件
    input_path = Path(INPUT_DIR)
    all_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.xlsx"))

    if not all_files:
        print(f"无文件： {INPUT_DIR}")
        sys.exit(0)

    # 用于存储所有读取的 DataFrame
    dataframes = []

    for file_path in sorted(all_files):  # sorted 保证处理顺序可预测
        try:
            print(f"正在读取 {file_path.name}")
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, skiprows=skiprows, dtype=str)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, skiprows=skiprows, dtype=str)
            else:
                continue  # 非目标格式跳过

            # 可选：记录来源文件名
            df["source_file"] = file_path.name
            dataframes.append(df)

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # 拼接所有数据
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df = data_cleaning.clear_invalid_chars(merged_df)  # 清洗
        merged_df.to_csv(merged_output, index=False, encoding="utf-8-sig")
        print("保存成功")
        # 清空beforeMerged目录
        for file_path in all_files:
            os.remove(file_path)
        print("清空beforeMerged目录成功")
    else:
        print("无数据")
