import pandas as pd
import os
import sys
import numpy as np

# 添加项目根目录到 sys.path
project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root_dir)

DATA_DIR = os.path.join(project_root_dir, "data", "works_lpdr", "correlationAnalysis")

from src.dataProcessing import data_cleaning
from src.analyzeMethods import statistical_methods
from src.analyzeMethods import statistical_tests
from src.analyzeMethods import clustering_methods

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(DATA_DIR, "data.xlsx"))
    df = data_cleaning.clear_invalid_chars(df)
    df = data_cleaning.convert_types(df)
    statistical_methods.correlation_analysis(df, "门店编号")
    statistical_tests.normality_test(df["客单数"])
    res = clustering_methods.kmeans_analysis(df, 3, ["客单数", "客单价"])
    print(res)
