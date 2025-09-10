"""
data_cleaning.py
提供常用的数据清洗方法
"""

import pandas as pd
from typing import Union
import numpy as np


# 统计缺失值
def count_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计指定数据框中每列的缺失值数量
    :param df: 输入数据框
    :return: 缺失值统计结果
    """
    return df.isnull().sum()


# 删除某列/所有列的空值对应的行
def drop_na(df: pd.DataFrame, columns: Union[str, list] = None) -> pd.DataFrame:
    """
    删除指定列或所有列的空值对应的行
    :param df: 输入数据框
    :param columns: 指定列名或列名列表，默认 None 表示所有列
    :return: 删除空值后的数据框
    """
    if columns is None:
        return df.dropna()
    elif isinstance(columns, str):
        return df.dropna(subset=[columns])
    elif isinstance(columns, list):
        return df.dropna(subset=columns)
    else:
        raise ValueError("columns 参数必须是字符串或字符串列表")


# 填充空值，method 可选 均值填充，上一个值填充，下一个值填充，指定值填充，中位数填充
def fill_na(
    df: pd.DataFrame, columns: Union[str, list] = None, method: str = "mean"
) -> pd.DataFrame:
    """
    填充指定列或所有列的空值
    :param df: 输入数据框
    :param columns: 指定列名或列名列表，默认 None 表示所有列
    :param method: 填充方法，默认为 'mean'，可选 'mean', 'ffill', 'bfill', 'value'
    :return: 填充空值后的数据框
    """
    if columns is None:
        if method == "mean":
            return df.fillna(df.mean(numeric_only=True))
        elif method == "ffill":
            return df.fillna(method="ffill")
        elif method == "bfill":
            return df.fillna(method="bfill")
        elif method == "value":
            return df.fillna(0)
        else:
            raise ValueError("method 参数必须是 'mean', 'ffill', 'bfill', 'value'")


# 清除无效字符
def clear_invalid_chars(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入数据框，清除字符串列中的无效字符
    :param df: 输入数据框
    保留：汉字、字母、数字、常见中英文标点和基本符号。
    """

    # 1. 清洗列名：只保留汉字、字母、数字、下划线、连字符，其他替换为下划线
    col_pattern = r"[^\w\u4e00-\u9fff\-_%]+"
    cleaned_columns = (
        pd.Series(df.columns)
        .astype(str)
        .str.replace(col_pattern, "_", regex=True)
        .str.strip("_")
        .tolist()
    )
    df = df.copy()
    df.columns = cleaned_columns

    # 2. 定义要保留的字符（白名单），匹配不在其中的字符进行清除
    # 保留：字母、数字、下划线、汉字、空白、常见标点符号
    pattern = r'[^\w\u4e00-\u9fff\s\.,;!?，。；！？()（）【】\[\]\'"\-/]+'

    # 3. 找出所有字符串类型列
    str_cols = df.select_dtypes(include=["object", "string"]).columns

    # 4. 对每列进行向量化清洗
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(pattern, "", regex=True)
            # 将 '<NA>' 转回 NaN（可选）
            df[col] = df[col].replace("nan", pd.NA)

    return df


# 将能转变为数字的字符串转为数字，能转变为日期的字符串转为日期
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    将能转变为数值的字符串转为数值
    :param df: 输入数据框
    :return: 转换类型后的数据框
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except ValueError:
            pass
    return df
