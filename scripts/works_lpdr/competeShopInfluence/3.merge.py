import os
import sys


project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root_dir)

DATA_DIR = os.path.join(
    project_root_dir, "data", "works_lpdr", "competeShopInfluence", "results"
)

OUTPUT_DIR = os.path.join(
    project_root_dir, "data", "works_lpdr", "competeShopInfluence"
)


# 读取data/final_data下所有csv文件，新增一列，列名为文件名称，然后拼接到一个表中保存
import pandas as pd

files = os.listdir(DATA_DIR)
files = [file for file in files if file.endswith(".csv")]

dfs = []
for file in files:
    df = pd.read_csv(os.path.join(DATA_DIR, file))
    df["时间段标识"] = file
    dfs.append(df)

result = pd.concat(dfs, ignore_index=True)
result.to_csv(os.path.join(OUTPUT_DIR, "竞争门店开业后数据.csv"), index=False)
