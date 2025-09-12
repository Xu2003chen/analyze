"""
其他统计分析方法
包含：关联分析
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time

# 中文适配
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False


# ================== 购物篮分析，关联规则挖掘 ==================
def market_basket_analysis(
    df: pd.DataFrame,
    transaction_id_col: str,
    item_col: str,
    min_support: float = 0.01,
    min_confidence: float = 0.01,
    min_lift: float = 0.01,
    max_length: Optional[int] = None,
    min_item_frequency: int = 2,  # 商品最少出现次数
) -> Dict[str, Any]:
    """
    购物篮分析（关联规则挖掘）- 使用 FP-Growth 加速 + 数据预处理防卡顿

    参数:
        df: 交易数据
        transaction_id_col: 单号列名
        item_col: 商品/维度列名
        min_support: 最小支持度阈值（默认 0.01）
        min_confidence: 最小置信度阈值（默认 0.3）
        min_lift: 最小提升度阈值（默认 1.0）
        max_length: 频繁项集最大长度（可选）
        min_item_frequency: 商品最少出现在多少个订单中才保留（去噪）

    返回:
        dict: 包含频繁项集、关联规则、图表等结果
    """
    start_time = time.time()

    if transaction_id_col not in df.columns:
        raise ValueError(f"交易ID列 '{transaction_id_col}' 不存在于数据中。")
    if item_col not in df.columns:
        raise ValueError(f"商品列 '{item_col}' 不存在于数据中。")

    # 提取所需列并去空
    data = df[[transaction_id_col, item_col]].dropna().copy()

    # === 预处理 1: 过滤低频商品 ===
    print(data[transaction_id_col].nunique())
    item_freq = data.groupby(item_col)[transaction_id_col].nunique()
    frequent_items = item_freq[item_freq >= min_item_frequency].index
    data = data[data[item_col].isin(frequent_items)]
    print(
        f"过滤低频商品后：{data[transaction_id_col].nunique()} 个订单，{data[item_col].nunique()} 种商品"
    )

    # === 生成事务列表 ===
    transactions = data.groupby(transaction_id_col)[item_col].apply(list).tolist()
    print(f"共 {len(transactions)} 笔事务用于分析")

    if len(transactions) == 0:
        print("⚠️ 处理后无有效事务数据。")
        rules_df = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
        frequent_itemsets = pd.DataFrame(columns=["itemsets", "support"])
    else:
        # === 使用 TransactionEncoder 转换为独热编码 ===
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # === 使用 FP-Growth 替代 Apriori（更快更省内存）===
        frequent_itemsets = fpgrowth(
            df_encoded, min_support=min_support, use_colnames=True, max_len=max_length
        )

        if len(frequent_itemsets) == 0:
            print("⚠️ 未发现满足最小支持度的频繁项集。")
            rules_df = pd.DataFrame(
                columns=["antecedents", "consequents", "support", "confidence", "lift"]
            )
        else:
            # === 生成关联规则 ===
            rules = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=min_confidence
            )
            rules = rules[rules["lift"] >= min_lift]
            rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

            # 格式化前件和后件为字符串
            rules["antecedents"] = rules["antecedents"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )
            rules["consequents"] = rules["consequents"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )

            rules_df = rules[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ]

    # ------------------ 可视化 ------------------
    plt.figure(figsize=(10, 7))

    if len(rules_df) > 0:
        # 控制显示数量，防止图表过载
        display_rules = rules_df.head(20)  # 最多显示 top 20
        scatter = plt.scatter(
            display_rules["confidence"],
            display_rules["lift"],
            s=display_rules["support"] * 2000,  # 放大 support 显示
            alpha=0.6,
            color="tab:blue",
            edgecolors="k",
            linewidth=0.5,
        )
        plt.xlabel("置信度 (Confidence)")
        plt.ylabel("提升度 (Lift)")
        plt.title(
            f"关联规则分析\n"
            f"最小支持度={min_support}, 最小置信度={min_confidence}, 提升度≥{min_lift}\n"
            f"规则数: {len(rules_df)} (显示前20条)"
        )
        plt.grid(True, linestyle="--", alpha=0.5)

        # 添加标签（仅 top 5，防止重叠）
        top5 = display_rules.head(5)
        for _, row in top5.iterrows():
            plt.text(
                row["confidence"],
                row["lift"],
                f"{row['antecedents']} → {row['consequents']}",
                fontsize=7,
                ha="right",
                va="bottom",
                alpha=0.8,
            )
    else:
        plt.text(
            0.5, 0.5, "无满足条件的关联规则", fontsize=14, ha="center", va="center"
        )
        plt.axis("off")
        plt.title("关联规则：无结果")

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()

    # ------------------ 返回结果 ------------------
    execution_time = time.time() - start_time

    return {
        "方法": "购物篮分析 (FP-Growth 关联规则)",
        "参数": {
            "最小支持度": min_support,
            "最小置信度": min_confidence,
            "最小提升度": min_lift,
            "最大项集长度": max_length,
            "商品最低出现次数": min_item_frequency,
            "最终交易数": len(transactions),
            "唯一商品数": data[item_col].nunique(),
            "执行时间（秒）": round(execution_time, 2),
        },
        "频繁项集": (
            frequent_itemsets.to_dict("records") if len(frequent_itemsets) > 0 else []
        ),
        "关联规则": rules_df.to_dict("records"),
        "规则数量": len(rules_df),
        "图表": fig,
        # 注意：不再返回原始事务列表以节省内存
    }
