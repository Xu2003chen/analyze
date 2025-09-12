import pandas as pd
import os
import sys
import numpy as np

# 添加项目根目录到 sys.path
project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root_dir)

from src.analyzeMethods import other_methods
from src.analyzeMethods import clustering_methods
from src.dataProcessing import data_cleaning
from src.constantClass.getBaseInfo import getBaseInfo

DATA_DIR = os.path.join(project_root_dir, "data", "works_lpdr", "correlationAnalysis")
OUTPUT_DIR = os.path.join(project_root_dir, "data", "works_lpdr", "correlationAnalysis")

if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv(os.path.join(DATA_DIR, "零售.csv"), dtype={"单据号": str})
    df = data_cleaning.clear_invalid_chars(df)
    shopinfo = getBaseInfo()["productInfo"]

    print(df.shape)
    print(df["单据号"].nunique())
    res = other_methods.market_basket_analysis(
        df, "单据号", "商品名称", min_item_frequency=5
    )
    print(res)
