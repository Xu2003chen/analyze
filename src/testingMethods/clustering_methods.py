"""
聚类模块：提供常用聚类方法
支持 KMeans、层次聚类、DBSCAN、GMM 等算法。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple , Literal
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 中文适配
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False


def _prepare_data(
    df: pd.DataFrame, features: Optional[list] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """预处理数据：标准化"""
    if features:
        data = df[features]
    else:
        data = df.select_dtypes(include=[np.number])

    # 去除缺失值
    data = data.dropna()

    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# ==================== KMeans 聚类 =====================
def kmeans_analysis(
    df: pd.DataFrame,
    n_clusters: int = 3,
    features: Optional[list] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    KMeans 聚类分析

    参数:
        df: 数据 DataFrame
        n_clusters: 聚类数
        features: 用于聚类的特征列名列表（默认为所有数值列）
        random_state: 随机种子

    返回:
        dict: 包含模型、标签、轮廓系数、图表等
    """
    scaled_data, scaler = _prepare_data(df, features)

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(scaled_data)

    sil_score = silhouette_score(scaled_data, labels)

    # 可视化（PCA降维到2D）
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Set1", s=50)
    plt.title(f"KMeans 聚类结果 (k={n_clusters})\n轮廓系数: {sil_score:.3f}")
    plt.xlabel("PCA 第一主成分")
    plt.ylabel("PCA 第二主成分")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "KMeans 聚类",
        "聚类数": n_clusters,
        "标签": labels,
        "轮廓系数": round(sil_score, 4),
        "模型": model,
        "PCA_坐标": coords,
        "图表": fig,
        "样本量": len(scaled_data),
    }


# ==================== 层次聚类 =====================
def hierarchical_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    features: Optional[list] = None,
) -> Dict[str, Any]:
    """
    层次聚类分析（Agglomerative）

    参数:
        df: 数据 DataFrame
        n_clusters: 聚类数
        linkage: 连接方式 ("ward", "complete", "average", "single")
        features: 特征列名列表

    返回:
        dict: 包含模型、标签、图表等
    """
    scaled_data, scaler = _prepare_data(df, features)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(scaled_data)

    sil_score = silhouette_score(scaled_data, labels)

    # 可视化
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=50)
    plt.title(f"层次聚类 (linkage={linkage})\n轮廓系数: {sil_score:.3f}")
    plt.xlabel("PCA 第一主成分")
    plt.ylabel("PCA 第二主成分")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "层次聚类",
        "聚类数": n_clusters,
        "连接方式": linkage,
        "标签": labels,
        "轮廓系数": round(sil_score, 4),
        "模型": model,
        "PCA_坐标": coords,
        "图表": fig,
        "样本量": len(scaled_data),
    }


# ==================== DBSCAN 聚类 =====================
def dbscan_analysis(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    features: Optional[list] = None,
) -> Dict[str, Any]:
    """
    DBSCAN 聚类分析

    参数:
        df: 数据 DataFrame
        eps: 邻域半径
        min_samples: 成为核心对象所需的最小样本数
        features: 特征列名列表

    返回:
        dict: 包含模型、标签、噪声点数、图表等
    """
    scaled_data, scaler = _prepare_data(df, features)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_data)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    sil_score = None
    if n_clusters > 1:
        sil_score = silhouette_score(scaled_data, labels)

    # 可视化
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # 黑色表示噪声
            marker = "x"
        else:
            marker = "o"
        class_member_mask = labels == k
        xy = coords[class_member_mask]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            c=[col],
            marker=marker,
            s=50,
            label=f"簇 {k}" if k != -1 else "噪声",
        )

    plt.title(f"DBSCAN 聚类结果\n簇数: {n_clusters}, 噪声点数: {n_noise}")
    plt.xlabel("PCA 第一主成分")
    plt.ylabel("PCA 第二主成分")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "DBSCAN 聚类",
        "eps": eps,
        "min_samples": min_samples,
        "标签": labels,
        "簇数": n_clusters,
        "噪声点数": n_noise,
        "轮廓系数": round(sil_score, 4) if sil_score else None,
        "模型": model,
        "PCA_坐标": coords,
        "图表": fig,
        "样本量": len(scaled_data),
    }


# ==================== 高斯混合模型 =====================
def gaussian_mixture_model(
    df: pd.DataFrame,
    n_components: int = 3,
    features: Optional[list] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    高斯混合模型（GMM）聚类分析

    参数:
        df: 数据 DataFrame
        n_components: 成分（簇）数
        features: 特征列名列表
        random_state: 随机种子

    返回:
        dict: 包含模型、标签、概率、图表等
    """
    scaled_data, scaler = _prepare_data(df, features)

    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)

    sil_score = silhouette_score(scaled_data, labels)

    # 可视化
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Set2", s=50)
    plt.title(f"GMM 聚类结果 (components={n_components})\n轮廓系数: {sil_score:.3f}")
    plt.xlabel("PCA 第一主成分")
    plt.ylabel("PCA 第二主成分")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "高斯混合模型 (GMM)",
        "成分数": n_components,
        "标签": labels,
        "概率": probabilities,
        "轮廓系数": round(sil_score, 4),
        "模型": model,
        "PCA_坐标": coords,
        "图表": fig,
        "样本量": len(scaled_data),
    }


# ==================== KMeans 最优聚类数 =====================
def find_optimal_kmeans_k(
    df: pd.DataFrame,
    k_range: range = range(2, 11),
    features: Optional[list] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    使用肘部法寻找 KMeans 最优聚类数 k

    参数:
        df: 数据 DataFrame
        k_range: 尝试的 k 值范围
        features: 特征列名列表
        random_state: 随机种子

    返回:
        dict: 包含 k 值列表、惯量列表、图表等
    """
    scaled_data, scaler = _prepare_data(df, features)

    inertias = []
    sil_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(scaled_data, kmeans.labels_))

    # 绘制肘部图
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("聚类数 k")
    ax1.set_ylabel("惯量 (Inertia)", color=color)
    ax1.plot(k_range, inertias, marker="o", color=color, label="惯量")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("轮廓系数", color=color)
    ax2.plot(k_range, sil_scores, marker="s", color=color, label="轮廓系数")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("KMeans 最优 k 值选择（肘部法 + 轮廓系数）")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

    fig = plt.gcf()

    return {
        "方法": "KMeans 最优 k 选择",
        "k值范围": list(k_range),
        "惯量列表": [round(i, 2) for i in inertias],
        "轮廓系数列表": [round(s, 4) for s in sil_scores],
        "图表": fig,
    }
