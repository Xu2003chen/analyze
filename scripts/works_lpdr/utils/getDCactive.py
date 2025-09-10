import pandas as pd
import os
import sys
import numpy as np

# 添加项目根目录到 sys.path
project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root_dir)

DATA_DIR = os.path.join(project_root_dir, "data", "works_lpdr", "utils")


df = pd.read_csv(os.path.join(DATA_DIR, "dc.csv"))
df = df.groupby("调往门店名称")["商品名称"].nunique().reset_index()

# 打印
print(df)
df.to_csv(os.path.join(DATA_DIR, "dc_active.csv"), index=False)
