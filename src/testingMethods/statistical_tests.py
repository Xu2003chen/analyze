"""
statistical_tests.py
提供常用的统计检验函数，便于在分析和测试中调用。
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 中文适配
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 方差齐性检验（F 检验） =====================
def f_test(
    group1: Union[list, np.ndarray, pd.Series],
    group2: Union[list, np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    方差齐性检验（F 检验）
    参数:
        group1: 第一组数据
        group2: 第二组数据
    返回:
        dict: 包含检验名称、F值、p值、是否显著、自由度、方差等信息
    """
    group1 = np.asarray(group1).astype(float)
    group2 = np.asarray(group2).astype(float)

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    if var1 >= var2:
        f_stat = var1 / var2
        dfn = len(group1) - 1
        dfd = len(group2) - 1
    else:
        f_stat = var2 / var1
        dfn = len(group2) - 1
        dfd = len(group1) - 1

    p_value = 2 * min(stats.f.cdf(f_stat, dfn, dfd), 1 - stats.f.cdf(f_stat, dfn, dfd))

    return {
        "检验": "F-test（方差齐性检验）",
        "f_statistic": float(f_stat),
        "p值": float(p_value),
        "significant": bool(p_value < 0.05),
        "自由度分子": int(dfn),
        "自由度分母": int(dfd),
        "方差1": float(var1),
        "方差2": float(var2),
    }


# ===================== t 检验 =====================
def t_test(
    group1: Union[list, np.ndarray, pd.Series],
    group2: Union[list, np.ndarray, pd.Series],
    paired: Union[bool, str] = "False",
) -> Dict[str, Any]:
    """
    统一的两样本 t 检验函数（独立或配对）

    参数:
        group1: 第一组数据
        group2: 第二组数据
        paired:
            - True: 使用配对检验
            - False: 独立样本检验

    返回:
        dict: 包含检验名称、统计量、p值、显著性等信息
    """
    group1 = np.asarray(group1).astype(float)
    group2 = np.asarray(group2).astype(float)

    n1, n2 = len(group1), len(group2)

    is_paired = bool(paired)
    if is_paired and n1 != n2:
        raise ValueError("配对样本的两组数据长度必须相等")
    # 执行检验
    if is_paired:
        stat, p = stats.ttest_rel(group1, group2, nan_policy="omit")
        df = n1 - 1
        diff = float(np.mean(group1 - group2))
        test_name = "Paired t-test（配对样本t检验）"
        extra_info = {"均值差": diff, "样本量": int(n1)}
    else:
        stat, p = stats.ttest_ind(group1, group2, nan_policy="omit", equal_var=False)
        df = n1 + n2 - 2  # Welch's t 的自由度是近似的，这里简化显示
        test_name = "Independent t-test（独立样本t检验）"
        extra_info = {
            "均值1": float(np.mean(group1)),
            "均值2": float(np.mean(group2)),
            "样本量1": int(n1),
            "样本量2": int(n2),
        }

    return {
        "检验": test_name,
        "t_statistic": float(stat),
        "p值": float(p),
        "significant": bool(p < 0.05),
        "自由度": int(df),
        **extra_info,
    }


# ===================== 卡方检验 =====================
def chi_square_test(observed: Union[list, np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    卡方检验（独立性或适合性检验）
    参数:
        observed: 观测频数表（2D 数组或 DataFrame）
    返回:
        dict: 包含检验名称、卡方值、p值、自由度、是否显著、期望频数
    """
    if isinstance(observed, pd.DataFrame):
        observed = observed.values
    else:
        observed = np.array(observed)

    stat, p, dof, expected = stats.chi2_contingency(observed)

    return {
        "检验": "Chi-square test（卡方检验）",
        "chi2": float(stat),
        "p值": float(p),
        "significant": bool(p < 0.05),
        "自由度": int(dof),
        "期望频数": expected.tolist(),  # 转为 list 便于序列化
        "观测频数": observed.tolist(),
    }


# ===================== 两样本比例 z 检验 =====================
def proportion_z_test(count: list, nobs: list) -> Dict[str, Any]:
    """
    两样本比例 z 检验
    参数:
        count: 每组中成功次数，如 [50, 60]
        nobs: 每组总样本数，如 [100, 100]
    返回:
        dict: 包含检验名称、z值、p值、是否显著、比例等信息
    """
    try:
        from statsmodels.stats.proportion import proportions_ztest
    except ImportError:
        raise ImportError("请安装 statsmodels: pip install statsmodels")

    count = np.array(count)
    nobs = np.array(nobs)

    stat, p = proportions_ztest(count, nobs)
    prop1 = count[0] / nobs[0]
    prop2 = count[1] / nobs[1]

    return {
        "检验": "Z-test for two proportions（两样本比例z检验）",
        "z_statistic": float(stat),
        "p值": float(p),
        "significant": bool(p < 0.05),
        "比例1": float(prop1),
        "比例2": float(prop2),
        "成功数1": int(count[0]),
        "成功数2": int(count[1]),
        "样本量1": int(nobs[0]),
        "样本量2": int(nobs[1]),
    }


# ===================== 正态检验 =====================
def normality_test(group: Union[list, np.ndarray, pd.Series]) -> Dict[str, Any]:
    """
    正态性检验（Shapiro-Wilk 检验）
    参数:
        group: 输入数据
    返回:
        六张图片，分别是原始数据、平方根变换、对数变换下的直方图和 Q-Q 图
        dict: 包含p
    """
    group = np.asarray(group).astype(float)

    original = group
    transformed_sqrt = np.sqrt(group)
    transformed_log = np.log1p(group)

    # 创建画布：2 行 3 列
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('不同变换下 "8月销量" 的分布对比', fontsize=16)

    # 标签
    names = ["原始数据", "平方根变换", "对数变换"]
    data_list = [original, transformed_sqrt, transformed_log]

    # 第一行：直方图 + KDE
    for i, (data, name) in enumerate(zip(data_list, names)):
        sns.histplot(data, kde=True, ax=axes[0, i])
        axes[0, i].set_title(f"直方图: {name}")
        axes[0, i].set_xlabel("值")

    # 第二行：Q-Q 图
    for i, (data, name) in enumerate(zip(data_list, names)):
        stats.probplot(data, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f"Q-Q 图: {name}")
        axes[1, i].set_xlabel("理论分位数")
        axes[1, i].set_ylabel("样本分位数")

    # 调整布局
    plt.tight_layout()
    plt.show()

    # 输出三种p值
    p_values = [float(stats.shapiro(data)[1]) for data in data_list]
    return {"检验": "Shapiro-Wilk test（正态性检验）", "p值": p_values}
