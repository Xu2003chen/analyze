import pandas as pd
import numpy as np

# 1. 读取数据
data = pd.read_csv("data/works_lpdr/competeShopInfluence/data.csv")

# 关键：将营业额、客单数等字段转为数值类型，非数字转为 NaN
numeric_columns = [
    "营业额",
    "客单数",
    "客单价",
    "会员商品销售金额",
    "会员客单数",
    "会员客单价",
]
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")  # 非法值转为 NaN

# 删除营业额小于 10 的行
data = data[data["营业额"] >= 10]

# 去重：门店名称和营业日都相同的行
data = data.drop_duplicates(subset=["门店名称", "营业日"])


# 保存清洗后的数据
data.to_csv(
    "data/works_lpdr/competeShopInfluence/datafinal.csv",
    index=False,
    encoding="utf-8-sig",
)
