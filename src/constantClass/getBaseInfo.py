# 读取常用表格，门店督导对应表

import os
import pandas as pd
import sys

project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root_dir)


def getBaseInfo():
    # 读取data/works_lpdr/.basedata下所有表格并返回
    # 返回字典，键为文件名，值为dataframe
    baseInfo = {}
    for file in os.listdir("data/works_lpdr/.base_data"):
        if file.endswith(".xlsx"):
            baseInfo[file.split(".")[0]] = pd.read_excel(
                "data/works_lpdr/.base_data/" + file
            )
    # 打印所有键
    print("键名:", list(baseInfo.keys()))
    return baseInfo


# shopinfo = getBaseInfo()["shopInfo copy"]
