# regression_analysis.py
"""
回归分析模块：提供常用的回归方法，便于数据分析和建模调用。
支持线性回归、岭回归、Lasso回归、决策树回归、随机森林回归等。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 中文适配
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False


def _prepare_data(
    df: pd.DataFrame, target_col: str, feature_cols: Optional[list] = None
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler
]:
    """预处理数据：分离特征和目标，划分训练集/测试集，标准化"""
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据中。")

    if feature_cols:
        feature_data = df[feature_cols]
    else:
        # 自动选择数值型特征列，排除目标列
        feature_data = df.select_dtypes(include=[np.number]).drop(
            columns=[target_col], errors="ignore"
        )

    target_data = df[target_col]

    # 去除缺失值
    combined = pd.concat([feature_data, target_data], axis=1).dropna()
    feature_data = combined[feature_data.columns]
    target_data = combined[target_col]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target_data, test_size=0.2, random_state=42
    )

    # 标准化特征
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 标准化目标（对某些模型如SVR有帮助）
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    )


# ===================== 线性回归 =====================
def linear_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    线性回归分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        feature_cols: 特征列名列表（默认为所有数值列，除去目标列）

    返回:
        dict: 包含模型、预测值、评估指标、图表等
    """
    X_train, X_test, y_train, y_test, _, _, scaler_X, _ = _prepare_data(
        df, target_col, feature_cols
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 可视化：真实值 vs 预测值
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"线性回归: 真实值 vs 预测值\nR² = {r2:.4f}, RMSE = {rmse:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "线性回归",
        "模型": model,
        "预测值": y_pred,
        "真实值": y_test.values,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "图表": fig,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
    }


# ===================== 岭回归 =====================
def ridge_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    alpha: float = 1.0,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    岭回归分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        alpha: 正则化强度
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    X_train, X_test, y_train, y_test, _, _, scaler_X, _ = _prepare_data(
        df, target_col, feature_cols
    )

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="orange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"岭回归 (α={alpha}): 真实值 vs 预测值\nR² = {r2:.4f}, RMSE = {rmse:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "岭回归",
        "正则化参数alpha": alpha,
        "模型": model,
        "预测值": y_pred,
        "真实值": y_test.values,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "图表": fig,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
    }


# ===================== Lasso回归 =====================
def lasso_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    alpha: float = 1.0,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Lasso回归分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        alpha: 正则化强度
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标、非零特征等
    """
    X_train, X_test, y_train, y_test, _, _, scaler_X, _ = _prepare_data(
        df, target_col, feature_cols
    )

    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 获取非零系数的特征
    coef = model.coef_
    feature_names = (
        df[feature_cols].columns
        if feature_cols
        else df.select_dtypes(include=[np.number])
        .drop(columns=[target_col], errors="ignore")
        .columns
    )
    selected_features = feature_names[coef != 0].tolist()

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="green")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(
        f"Lasso回归 (α={alpha}): 真实值 vs 预测值\nR² = {r2:.4f}, RMSE = {rmse:.4f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "Lasso回归",
        "正则化参数alpha": alpha,
        "模型": model,
        "预测值": y_pred,
        "真实值": y_test.values,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "非零特征": selected_features,
        "图表": fig,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
    }


# ===================== 决策树回归 =====================
def decision_tree_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    max_depth: Optional[int] = None,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    决策树回归分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        max_depth: 树的最大深度
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    X_train, X_test, y_train, y_test, _, _, scaler_X, _ = _prepare_data(
        df, target_col, feature_cols
    )

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="red")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(
        f"决策树回归 (max_depth={max_depth}): 真实值 vs 预测值\nR² = {r2:.4f}, RMSE = {rmse:.4f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "决策树回归",
        "最大深度": max_depth,
        "模型": model,
        "预测值": y_pred,
        "真实值": y_test.values,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "图表": fig,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
    }


# ===================== 随机森林回归 =====================
def random_forest_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    随机森林回归分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        n_estimators: 树的数量
        max_depth: 树的最大深度
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    X_train, X_test, y_train, y_test, _, _, scaler_X, _ = _prepare_data(
        df, target_col, feature_cols
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="purple")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(
        f"随机森林回归 (trees={n_estimators}): 真实值 vs 预测值\nR² = {r2:.4f}, RMSE = {rmse:.4f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "随机森林回归",
        "树数量": n_estimators,
        "最大深度": max_depth,
        "模型": model,
        "预测值": y_pred,
        "真实值": y_test.values,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "图表": fig,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
    }


# ===================== 支持向量回归 =====================
def support_vector_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    支持向量回归 (SVR) 分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        kernel: 核函数 ('linear', 'poly', 'rbf', 'sigmoid')
        C: 正则化参数
        gamma: 核函数系数
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    ) = _prepare_data(df, target_col, feature_cols)

    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train_scaled)  # SVR通常对标准化数据效果更好
    y_pred_scaled = model.predict(X_test)
    # 反标准化预测结果
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="brown")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(
        f"支持向量回归 (kernel={kernel}): 真实值 vs 预测值\nR² = {r2:.4f}, RMSE = {rmse:.4f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "支持向量回归 (SVR)",
        "核函数": kernel,
        "正则化参数C": C,
        "gamma": gamma,
        "模型": model,
        "预测值": y_pred,
        "真实值": y_test.values,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "图表": fig,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
    }
