"""
statistical_tests.py
提供常用的统计检验方法，便于在分析和测试中调用。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Union

# 中文适配
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 单因素方差分析 =====================
def one_way_anova(data, value_col, group_col):
    """
    单因素方差分析（One-way ANOVA）
    用于检验一个分类变量（因素）对连续因变量的影响。

    参数:
        data (pd.DataFrame): 包含数据的 DataFrame
        value_col (str): 因变量列名（如 '8月销量'）
        group_col (str): 分组变量列名（如 '区域'）

    返回:
        dict: 包含 F 值、p 值及是否显著的结果字典
    """
    from scipy.stats import f_oneway

    # 提取数据，去除缺失值
    df_clean = data[[value_col, group_col]].dropna()

    # 检查是否有至少两个组
    if df_clean[group_col].nunique() < 2:
        raise ValueError(f"分组变量 '{group_col}' 至少需要包含 2 个不同的组")

    # 按分组变量分组提取数值
    groups = [group[value_col].values for name, group in df_clean.groupby(group_col)]

    # 执行单因素方差分析
    f_stat, p_val = f_oneway(*groups)

    return {
        "检验": "单因素方差分析（One-way ANOVA）",
        "因变量": value_col,
        "分组变量": group_col,
        f"{group_col}_F": round(f_stat, 4),
        f"{group_col}_p值": round(p_val, 6),
        f"{group_col}_显著": p_val < 0.05,
    }


# ===================== 双因素方差分析 =====================
def two_way_anova(data, value_col, factor_a, factor_b):
    """
    双因素方差分析（Two-way ANOVA）
    用于检验两个分类变量（因素）及其交互作用对连续因变量的影响。
    参数:
        data (pd.DataFrame): 包含数据的 DataFrame
        value_col (str): 因变量列名（如 '8月销量'）
        factor_a (str): 第一个分组变量（因素A，如 '区域'）
        factor_b (str): 第二个分组变量（因素B，如 '门店类型'）
    返回:
        dict: 包含主效应和交互效应的 F 值、p 值及是否显著的结果字典
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df_clean = data[[value_col, factor_a, factor_b]]

    # 检查是否有至少两个组（每个因素）
    if df_clean[factor_a].nunique() < 2:
        raise ValueError(f"因素 '{factor_a}' 至少需要包含 2 个不同的组")
    if df_clean[factor_b].nunique() < 2:
        raise ValueError(f"因素 '{factor_b}' 至少需要包含 2 个不同的组")

    # 构建模型公式：包含主效应和交互项
    formula = (
        f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})"
    )
    model = ols(formula, data=df_clean).fit()
    anova_table = anova_lm(model, typ=2)  # 使用 Type II 平方和（适用于平衡设计）

    # 提取结果
    def get_value(anova_table, key, col):
        return anova_table.loc[key, col] if key in anova_table.index else None

    # 获取各项的 F 和 p 值
    f_a = get_value(anova_table, f"C({factor_a})", "F")
    p_a = get_value(anova_table, f"C({factor_a})", "PR(>F)")

    f_b = get_value(anova_table, f"C({factor_b})", "F")
    p_b = get_value(anova_table, f"C({factor_b})", "PR(>F)")

    f_interaction = get_value(anova_table, f"C({factor_a}):C({factor_b})", "F")
    p_interaction = get_value(anova_table, f"C({factor_a}):C({factor_b})", "PR(>F)")

    return {
        "检验": "双因素方差分析（Two-way ANOVA）",
        "因变量": value_col,
        "因素A": factor_a,
        "因素B": factor_b,
        f"{factor_a}_F": round(f_a, 4) if f_a else None,
        f"{factor_a}_p值": round(p_a, 6) if p_a else None,
        f"{factor_a}_显著": bool(p_a < 0.05) if p_a else False,
        f"{factor_b}_F": round(f_b, 4) if f_b else None,
        f"{factor_b}_p值": round(p_b, 6) if p_b else None,
        f"{factor_b}_显著": bool(p_b < 0.05) if p_b else False,
        "交互作用_F": round(f_interaction, 4) if f_interaction else None,
        "交互作用_p值": round(p_interaction, 6) if p_interaction else None,
        "交互作用显著": bool(p_interaction < 0.05) if p_interaction else False,
    }


# ===================== 相关性分析 =====================
def correlation_analysis(
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """
    相关性分析：计算数值变量与目标变量的相关性，生成相关性热力图

    参数:
        df: 包含数据的 DataFrame
        target_col: 因变量（目标变量）的列名

    返回:
        dict: 包含检验名称、相关性矩阵、目标变量相关性、是否显著（仅提示）、图表对象等信息
    """
    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"列 '{target_col}' 不存在于 DataFrame 中。")

    # 仅选择数值型列
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        raise ValueError(f"目标列 '{target_col}' 必须是数值型。")

    # 计算相关性矩阵
    corr_matrix = numeric_df.corr(method="pearson")
    target_corr = corr_matrix[[target_col]].sort_values(
        by=target_col, key=lambda x: abs(x), ascending=False
    )

    # 提取与目标变量的相关性
    other_corr = target_corr.drop(index=target_col)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"相关性热力图\n目标变量: {target_col}")
    plt.tight_layout()
    plt.show()

    # 获取当前图像对象
    fig = plt.gcf()

    # 返回结果字典
    return {
        "检验": "Pearson 相关性分析",
        "相关性矩阵": corr_matrix.round(4),
        "目标变量相关性排序": target_corr.round(4),
        "其他变量与目标相关性": other_corr.round(4),
        "p值显著性": None,  # 可扩展为计算p值
        "图表": fig,
        "样本量": len(numeric_df),
        "分析变量数": numeric_df.shape[1],
    }
