# classification_analysis.py
"""
分类分析模块：提供常用的分类方法，便于数据分析和建模调用。
支持逻辑回归、K近邻、决策树、随机森林、支持向量机等。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# 中文适配
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False


def _prepare_data(
    df: pd.DataFrame, target_col: str, feature_cols: Optional[list] = None
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder
]:
    """预处理数据：分离特征和目标，划分训练集/测试集，标准化，标签编码"""
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
        feature_data, target_data, test_size=0.2, random_state=42, stratify=target_data
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 标签编码 (处理字符串标签)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, le


def logistic_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list] = None,
    probability: bool = True,
) -> Dict[str, Any]:
    """
    逻辑回归分类分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        feature_cols: 特征列名列表（默认为所有数值列，除去目标列）
        probability: 是否返回预测概率

    返回:
        dict: 包含模型、预测值、评估指标、图表等
    """
    X_train, X_test, y_train, y_test, scaler, le = _prepare_data(
        df, target_col, feature_cols
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 获取预测概率
    if probability and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        # 对于多分类，计算宏平均AUC
        classes = np.unique(y_train)
        if len(classes) > 2:
            y_test_bin = label_binarize(y_test, classes=classes)
            try:
                roc_auc = roc_auc_score(
                    y_test_bin, y_proba, average="macro", multi_class="ovr"
                )
            except ValueError:
                roc_auc = None  # 可能因标签不足而无法计算
        else:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        y_proba = None
        roc_auc = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("逻辑回归 - 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()
    fig_cm = plt.gcf()

    # ROC曲线 (仅适用于二分类)
    fig_roc = None
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc_single = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc_single,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("假正率 (False Positive Rate)")
        plt.ylabel("真正率 (True Positive Rate)")
        plt.title("逻辑回归 - ROC 曲线")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        fig_roc = plt.gcf()

    return {
        "方法": "逻辑回归",
        "模型": model,
        "预测标签": le.inverse_transform(y_pred),
        "真实标签": le.inverse_transform(y_test),
        "预测概率": y_proba,
        "准确率": round(accuracy, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1分数": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
        "混淆矩阵": cm,
        "混淆矩阵图表": fig_cm,
        "ROC曲线图表": fig_roc,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
        "类别标签": le.classes_.tolist(),
    }


def knn_classification_analysis(
    df: pd.DataFrame,
    target_col: str,
    n_neighbors: int = 5,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    K近邻 (KNN) 分类分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        n_neighbors: 邻居数量
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    X_train, X_test, y_train, y_test, scaler, le = _prepare_data(
        df, target_col, feature_cols
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"KNN (k={n_neighbors}) - 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()
    fig_cm = plt.gcf()

    return {
        "方法": "K近邻 (KNN)",
        "邻居数": n_neighbors,
        "模型": model,
        "预测标签": le.inverse_transform(y_pred),
        "真实标签": le.inverse_transform(y_test),
        "准确率": round(accuracy, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1分数": round(f1, 4),
        "混淆矩阵": cm,
        "混淆矩阵图表": fig_cm,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
        "类别标签": le.classes_.tolist(),
    }


def decision_tree_classification_analysis(
    df: pd.DataFrame,
    target_col: str,
    max_depth: Optional[int] = None,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    决策树分类分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        max_depth: 树的最大深度
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    X_train, X_test, y_train, y_test, scaler, le = _prepare_data(
        df, target_col, feature_cols
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"决策树 (max_depth={max_depth}) - 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()
    fig_cm = plt.gcf()

    return {
        "方法": "决策树",
        "最大深度": max_depth,
        "模型": model,
        "预测标签": le.inverse_transform(y_pred),
        "真实标签": le.inverse_transform(y_test),
        "准确率": round(accuracy, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1分数": round(f1, 4),
        "混淆矩阵": cm,
        "混淆矩阵图表": fig_cm,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
        "类别标签": le.classes_.tolist(),
    }


def random_forest_classification_analysis(
    df: pd.DataFrame,
    target_col: str,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    随机森林分类分析

    参数:
        df: 数据 DataFrame
        target_col: 目标变量列名
        n_estimators: 树的数量
        max_depth: 树的最大深度
        feature_cols: 特征列名列表

    返回:
        dict: 包含模型、预测值、评估指标等
    """
    X_train, X_test, y_train, y_test, scaler, le = _prepare_data(
        df, target_col, feature_cols
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"随机森林 (trees={n_estimators}) - 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()
    fig_cm = plt.gcf()

    return {
        "方法": "随机森林",
        "树数量": n_estimators,
        "最大深度": max_depth,
        "模型": model,
        "预测标签": le.inverse_transform(y_pred),
        "真实标签": le.inverse_transform(y_test),
        "准确率": round(accuracy, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1分数": round(f1, 4),
        "混淆矩阵": cm,
        "混淆矩阵图表": fig_cm,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
        "类别标签": le.classes_.tolist(),
    }


def support_vector_classification_analysis(
    df: pd.DataFrame,
    target_col: str,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    支持向量机 (SVM) 分类分析

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
    X_train, X_test, y_train, y_test, scaler, le = _prepare_data(
        df, target_col, feature_cols
    )

    model = SVC(
        kernel=kernel, C=C, gamma=gamma, probability=True
    )  # 开启probability以计算AUC
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 获取预测概率
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    # 计算AUC
    classes = np.unique(y_train)
    if len(classes) > 2:
        y_test_bin = label_binarize(y_test, classes=classes)
        try:
            roc_auc = roc_auc_score(
                y_test_bin, y_proba, average="macro", multi_class="ovr"
            )
        except ValueError:
            roc_auc = None
    else:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"SVM (kernel={kernel}) - 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()
    fig_cm = plt.gcf()

    return {
        "方法": "支持向量机 (SVM)",
        "核函数": kernel,
        "正则化参数C": C,
        "gamma": gamma,
        "模型": model,
        "预测标签": le.inverse_transform(y_pred),
        "真实标签": le.inverse_transform(y_test),
        "预测概率": y_proba,
        "准确率": round(accuracy, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1分数": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
        "混淆矩阵": cm,
        "混淆矩阵图表": fig_cm,
        "特征数": X_train.shape[1],
        "样本量_训练集": len(y_train),
        "样本量_测试集": len(y_test),
        "类别标签": le.classes_.tolist(),
    }
