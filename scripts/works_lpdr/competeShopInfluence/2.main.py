import pandas as pd
from datetime import timedelta
import os

# 设置DT
D_values = [10000]
T_values = [0]

# 检查 D 和 T 列表长度是否一致
if len(D_values) != len(T_values):
    raise ValueError("D_values 和 T_values 列表的长度必须相同。")

print(f"准备开始批量分析，共 {len(D_values)} 组参数...")

# ========== 读取数据 (只需读取一次) ==========
print("正在加载数据...")
try:
    # 输入文件路径 (基于项目根目录)
    data_file_path = os.path.join(
        "data", "works_lpdr", "competeShopInfluence", "datafinal.csv"
    )
    shopvs_file_path = os.path.join(
        "data", "works_lpdr", "competeShopInfluence", "shopvs.xlsx"
    )

    data = pd.read_csv(data_file_path, dtype=str, encoding="utf-8")
    open_dates = pd.read_excel(shopvs_file_path, sheet_name=0)
except Exception as e:
    print(f"❌ 无法读取数据文件：\n{e}")
    input("按回车键退出...")
    exit()

# 数据清洗 (只需执行一次)
print("正在清洗数据...")
data["营业日"] = (
    data["营业日"]
    .str.strip()
    .str.replace(r"^(\d{8})\.0*$", r"\1", regex=True)
    .str.replace(r"[^\d]", "", regex=True)
)
data["营业日"] = pd.to_datetime(data["营业日"], format="%Y%m%d", errors="coerce")

open_dates["竞争店开业时间"] = pd.to_datetime(
    open_dates["竞争店开业时间"], errors="coerce"
)

# 数值列转换 (只需执行一次)
numeric_columns = ["营业额", "客单数", "会员商品销售金额", "会员客单数"]
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# ========== 计算每个门店的最早营业时间（视为开业时间）(只需执行一次) ==========
print("正在计算各门店最早营业时间（视为开业时间）...")
earliest_open_date = data.groupby("门店名称")["营业日"].min().reset_index()
earliest_open_date.rename(columns={"营业日": "门店最早营业时间"}, inplace=True)

# === 提取无竞争门店列表 (只需执行一次) ===
no_compete_stores = open_dates[open_dates["是否存在竞争"] == "否"]["门店名称"].tolist()
print(f"📌 无竞争门店数量: {len(no_compete_stores)}，用于‘无竞争门店店均’计算")
if len(no_compete_stores) == 0:
    print("⚠️  警告：未找到无竞争门店，‘无竞争门店店均’将为空")
else:
    print("无竞争门店列表:", ", ".join(no_compete_stores))

# ========== 循环开始 ==========
for i, (D, T) in enumerate(zip(D_values, T_values)):
    print(f"\n{'='*20} 开始执行第 {i+1} 组分析: D={D}, T={T} {'='*20}")

    print(f"参数已设置：D = {D}, T = {T}")
    print(f"开业前时间段：开业时间-{T + D}天 → 开业时间-{T}天")
    print(f"开业后时间段：开业时间+{T}天 → 开业时间+{T + D}天")
    print(f"开业后同比时间段：开业时间+{T}-365天 → 开业时间+{T + D}-365天")
    print("开始分析...")

    results = []

    # 只分析“存在竞争”的门店
    compete_df = open_dates[open_dates["是否存在竞争"] == "是"].copy()

    for _, row in compete_df.iterrows():
        store = row["门店名称"]
        open_date = row["竞争店开业时间"]

        store_data = data[data["门店名称"] == store].copy()
        if store_data.empty:
            print(f"🟡 门店 '{store}' 在 datafinal.csv 中无数据")
            continue

        # 定义三个区间（左闭右开 [start, end)）
        intervals = {
            "开业前时间段": (
                open_date - timedelta(days=T + D),
                open_date - timedelta(days=T),
            ),
            "开业后时间段": (
                open_date + timedelta(days=T),
                open_date + timedelta(days=T + D),
            ),
            "开业后同比时间段": (
                open_date + timedelta(days=T) - timedelta(days=365),
                open_date + timedelta(days=T + D) - timedelta(days=365),
            ),
        }

        result_row = {"门店名称": store}

        for label, (start, end) in intervals.items():
            # 1. 当前门店日均 + 营业天数（保留）
            mask = (store_data["营业日"] >= start) & (store_data["营业日"] < end)
            filtered = store_data.loc[mask].copy()

            n_days = len(filtered)
            result_row[f"本门店_{label}_营业天数"] = n_days

            if not filtered.empty:
                avg_sales = filtered["营业额"].mean()
                avg_transactions = filtered["客单数"].mean()
                avg_member_sales = filtered["会员商品销售金额"].mean()
                avg_member_transactions = filtered["会员客单数"].mean()

                avg_customer_price = (
                    avg_sales / avg_transactions if avg_transactions > 0 else 0.0
                )
                avg_member_price = (
                    avg_member_sales / avg_member_transactions
                    if avg_member_transactions > 0
                    else 0.0
                )
                sales_ratio = avg_member_sales / avg_sales if avg_sales > 0 else 0.0
                transaction_ratio = (
                    avg_member_transactions / avg_transactions
                    if avg_transactions > 0
                    else 0.0
                )

                result_row[f"本门店_{label}_日均营业额"] = avg_sales
                result_row[f"本门店_{label}_日均客单数"] = avg_transactions
                result_row[f"本门店_{label}_客单价"] = avg_customer_price
                result_row[f"本门店_{label}_日均会员商品销售金额"] = avg_member_sales
                result_row[f"本门店_{label}_日均会员客单数"] = avg_member_transactions
                result_row[f"本门店_{label}_会员客单价"] = avg_member_price
                result_row[f"本门店_{label}_会员商品销售金额占比"] = sales_ratio
                result_row[f"本门店_{label}_会员客单数占比"] = transaction_ratio
            else:
                cols = [
                    "日均营业额",
                    "日均客单数",
                    "客单价",
                    "日均会员商品销售金额",
                    "日均会员客单数",
                    "会员客单价",
                    "会员商品销售金额占比",
                    "会员客单数占比",
                ]
                for col in cols:
                    result_row[f"本门店_{label}_{col}"] = pd.NA

            # 2. 全门店全局日均
            all_store_mask = (data["营业日"] >= start) & (data["营业日"] < end)
            all_store_filtered = data.loc[all_store_mask]

            if not all_store_filtered.empty:
                daily_agg = all_store_filtered.groupby(["门店名称", "营业日"])[
                    numeric_columns
                ].sum()
                global_avg = daily_agg.mean()

                result_row[f"全门店_{label}_日均营业额"] = global_avg["营业额"]
                result_row[f"全门店_{label}_日均客单数"] = global_avg["客单数"]
                result_row[f"全门店_{label}_客单价"] = (
                    global_avg["营业额"] / global_avg["客单数"]
                    if global_avg["客单数"] > 0
                    else 0.0
                )
                result_row[f"全门店_{label}_日均会员商品销售金额"] = global_avg[
                    "会员商品销售金额"
                ]
                result_row[f"全门店_{label}_日均会员客单数"] = global_avg["会员客单数"]
                result_row[f"全门店_{label}_会员客单价"] = (
                    global_avg["会员商品销售金额"] / global_avg["会员客单数"]
                    if global_avg["会员客单数"] > 0
                    else 0.0
                )
                result_row[f"全门店_{label}_会员商品销售金额占比"] = (
                    global_avg["会员商品销售金额"] / global_avg["营业额"]
                    if global_avg["营业额"] > 0
                    else 0.0
                )
                result_row[f"全门店_{label}_会员客单数占比"] = (
                    global_avg["会员客单数"] / global_avg["客单数"]
                    if global_avg["客单数"] > 0
                    else 0.0
                )
            else:
                cols = [
                    "日均营业额",
                    "日均客单数",
                    "客单价",
                    "日均会员商品销售金额",
                    "日均会员客单数",
                    "会员客单价",
                    "会员商品销售金额占比",
                    "会员客单数占比",
                ]
                for col in cols:
                    result_row[f"全门店_{label}_{col}"] = pd.NA

            # 3. 无竞争门店全局日均
            if len(no_compete_stores) > 0:
                no_compete_data = data[data["门店名称"].isin(no_compete_stores)].copy()
                no_compete_mask = (no_compete_data["营业日"] >= start) & (
                    no_compete_data["营业日"] < end
                )
                no_compete_filtered = no_compete_data.loc[no_compete_mask]

                if not no_compete_filtered.empty:
                    daily_agg_nc = no_compete_filtered.groupby(["门店名称", "营业日"])[
                        numeric_columns
                    ].sum()
                    global_avg_nc = daily_agg_nc.mean()

                    result_row[f"无竞争门店_{label}_日均营业额"] = global_avg_nc[
                        "营业额"
                    ]
                    result_row[f"无竞争门店_{label}_日均客单数"] = global_avg_nc[
                        "客单数"
                    ]
                    result_row[f"无竞争门店_{label}_客单价"] = (
                        global_avg_nc["营业额"] / global_avg_nc["客单数"]
                        if global_avg_nc["客单数"] > 0
                        else 0.0
                    )
                    result_row[f"无竞争门店_{label}_日均会员商品销售金额"] = (
                        global_avg_nc["会员商品销售金额"]
                    )
                    result_row[f"无竞争门店_{label}_日均会员客单数"] = global_avg_nc[
                        "会员客单数"
                    ]
                    result_row[f"无竞争门店_{label}_会员客单价"] = (
                        global_avg_nc["会员商品销售金额"] / global_avg_nc["会员客单数"]
                        if global_avg_nc["会员客单数"] > 0
                        else 0.0
                    )
                    result_row[f"无竞争门店_{label}_会员商品销售金额占比"] = (
                        global_avg_nc["会员商品销售金额"] / global_avg_nc["营业额"]
                        if global_avg_nc["营业额"] > 0
                        else 0.0
                    )
                    result_row[f"无竞争门店_{label}_会员客单数占比"] = (
                        global_avg_nc["会员客单数"] / global_avg_nc["客单数"]
                        if global_avg_nc["客单数"] > 0
                        else 0.0
                    )
                else:
                    cols = [
                        "日均营业额",
                        "日均客单数",
                        "客单价",
                        "日均会员商品销售金额",
                        "日均会员客单数",
                        "会员客单价",
                        "会员商品销售金额占比",
                        "会员客单数占比",
                    ]
                    for col in cols:
                        result_row[f"无竞争门店_{label}_{col}"] = pd.NA
            else:
                cols = [
                    "日均营业额",
                    "日均客单数",
                    "客单价",
                    "日均会员商品销售金额",
                    "日均会员客单数",
                    "会员客单价",
                    "会员商品销售金额占比",
                    "会员客单数占比",
                ]
                for col in cols:
                    result_row[f"无竞争门店_{label}_{col}"] = pd.NA

        results.append(result_row)

    # ========== 输出结果 ==========
    result_df = pd.DataFrame(results)

    # ========== 合并为一个汇总表 ==========
    if not result_df.empty:
        # 基础信息 + 原始数据
        final_df = open_dates[open_dates["是否存在竞争"] == "是"][
            ["门店名称", "竞争店名称", "竞争店品牌", "竞争店距离", "竞争店开业时间"]
        ].copy()
        final_df = final_df.merge(result_df, on="门店名称", how="left")
        # 合并门店最早营业时间
        final_df = final_df.merge(earliest_open_date, on="门店名称", how="left")

        # 更新基础指标和比率指标列表
        metrics = ["日均营业额", "日均客单数", "日均会员商品销售金额", "日均会员客单数"]
        ratio_metrics = [
            "客单价",
            "会员客单价",
            "会员商品销售金额占比",
            "会员客单数占比",
        ]

        # 添加环比开业前
        for metric in metrics + ratio_metrics:
            # 本门店环比
            after_col = f"本门店_开业后时间段_{metric}"
            before_col = f"本门店_开业前时间段_{metric}"
            ratio_col = f"本门店_开业前时间段_环比{metric}"
            final_df[ratio_col] = (
                final_df[after_col] / final_df[before_col] - 1
            ).where(final_df[before_col] > 0, pd.NA)

            # 全门店环比
            all_after_col = f"全门店_开业后时间段_{metric}"
            all_before_col = f"全门店_开业前时间段_{metric}"
            all_ratio_col = f"全门店_开业前时间段_环比{metric}"
            final_df[all_ratio_col] = (
                final_df[all_after_col] / final_df[all_before_col] - 1
            ).where(final_df[all_before_col] > 0, pd.NA)

            # 无竞争门店环比
            nc_after_col = f"无竞争门店_开业后时间段_{metric}"
            nc_before_col = f"无竞争门店_开业前时间段_{metric}"
            nc_ratio_col = f"无竞争门店_开业前时间段_环比{metric}"
            final_df[nc_ratio_col] = (
                final_df[nc_after_col] / final_df[nc_before_col] - 1
            ).where(final_df[nc_before_col] > 0, pd.NA)

            # 差值（本门店环比 - 全门店环比）
            diff_col = f"差_本门店_全门店_开业前时间段_环比{metric}"
            final_df[diff_col] = final_df[ratio_col] - final_df[all_ratio_col]

            # 差值（本门店环比 - 无竞争门店环比）
            diff_col_nc = f"差_本门店_无竞争门店_开业前时间段_环比{metric}"
            final_df[diff_col_nc] = final_df[ratio_col] - final_df[nc_ratio_col]

        # 添加环比一年前
        for metric in metrics + ratio_metrics:
            # 本门店环比
            after_col = f"本门店_开业后时间段_{metric}"
            last_year_col = f"本门店_开业后同比时间段_{metric}"
            ratio_col = f"本门店_同比时间段_环比{metric}"
            final_df[ratio_col] = (
                final_df[after_col] / final_df[last_year_col] - 1
            ).where(final_df[last_year_col] > 0, pd.NA)

            # 全门店环比
            all_after_col = f"全门店_开业后时间段_{metric}"
            all_last_year_col = f"全门店_开业后同比时间段_{metric}"
            all_ratio_col = f"全门店_同比时间段_环比{metric}"
            final_df[all_ratio_col] = (
                final_df[all_after_col] / final_df[all_last_year_col] - 1
            ).where(final_df[all_last_year_col] > 0, pd.NA)

            # 无竞争门店环比
            nc_after_col = f"无竞争门店_开业后时间段_{metric}"
            nc_last_year_col = f"无竞争门店_开业后同比时间段_{metric}"
            nc_ratio_col = f"无竞争门店_同比时间段_环比{metric}"
            final_df[nc_ratio_col] = (
                final_df[nc_after_col] / final_df[nc_last_year_col] - 1
            ).where(final_df[nc_last_year_col] > 0, pd.NA)

            # 差值（本门店环比 - 全门店环比）
            diff_col = f"差_本门店_全门店_同比时间段_环比{metric}"
            final_df[diff_col] = final_df[ratio_col] - final_df[all_ratio_col]

            # 差值（本门店环比 - 无竞争门店环比）
            diff_col_nc = f"差_本门店_无竞争门店_同比时间段_环比{metric}"
            final_df[diff_col_nc] = final_df[ratio_col] - final_df[nc_ratio_col]

        # 列排序
        columns_order = [
            "门店名称",
            "竞争店名称",
            "竞争店品牌",
            "竞争店距离",
            "竞争店开业时间",
            "门店最早营业时间",
        ]

        period_labels = ["开业前时间段", "开业后时间段", "开业后同比时间段"]
        base_metrics = [
            "日均营业额",
            "日均客单数",
            "客单价",
            "日均会员商品销售金额",
            "日均会员客单数",
            "会员客单价",
            "会员商品销售金额占比",
            "会员客单数占比",
        ]

        # 添加基础数据列
        for label in period_labels:
            # 本门店基础数据
            columns_order.append(f"本门店_{label}_营业天数")
            for m in base_metrics:
                columns_order.append(f"本门店_{label}_{m}")
            # 全门店基础数据
            for m in base_metrics:
                columns_order.append(f"全门店_{label}_{m}")
            # 无竞争门店基础数据
            for m in base_metrics:
                columns_order.append(f"无竞争门店_{label}_{m}")

        # 添加环比数据列
        # 环比开业前
        for metric in metrics + ratio_metrics:
            columns_order.append(f"本门店_开业前时间段_环比{metric}")
            columns_order.append(f"全门店_开业前时间段_环比{metric}")
            columns_order.append(f"无竞争门店_开业前时间段_环比{metric}")
            columns_order.append(f"差_本门店_全门店_开业前时间段_环比{metric}")
            columns_order.append(f"差_本门店_无竞争门店_开业前时间段_环比{metric}")

        # 环比一年前 (同比时间段)
        for metric in metrics + ratio_metrics:
            columns_order.append(f"本门店_同比时间段_环比{metric}")
            columns_order.append(f"全门店_同比时间段_环比{metric}")
            columns_order.append(f"无竞争门店_同比时间段_环比{metric}")
            columns_order.append(f"差_本门店_全门店_同比时间段_环比{metric}")
            columns_order.append(f"差_本门店_无竞争门店_同比时间段_环比{metric}")

        # 确保所有需要的列都在 DataFrame 中，不存在的列用 NA 填充
        for col in columns_order:
            if col not in final_df.columns:
                final_df[col] = pd.NA

        final_df = final_df[columns_order]

        # 格式化
        for col in final_df.columns:
            if col.endswith(
                (
                    "_环比营业额",
                    "_环比客单数",
                    "_环比会员商品销售金额",
                    "_环比会员客单数",
                    "_环比客单价",
                    "_环比会员客单价",
                    "_环比会员商品销售金额占比",
                    "_环比会员客单数占比",
                )
            ):
                final_df[col] = final_df[col].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "NA"
                )
            elif col.startswith("差_"):
                final_df[col] = final_df[col].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
                )
            elif col.endswith("_营业天数"):
                final_df[col] = final_df[col].map(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "NA"
                )

        # 保存为一个汇总文件，文件名包含当前的 D 和 T 参数
        # 输出文件路径 (基于项目根目录)
        output_dir = os.path.join(
            "data", "works_lpdr", "competeShopInfluence", "results"
        )
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

        filename_summary = f"开业前时间段：开业时间-{T+D}天→开业时间-{T}天；开业后时间段：开业时间+{T}天→开业时间+{T+D}天；开业后同比时间段：开业时间+{T}-365天→开业时间+{T+D}-365天_汇总表.csv"
        full_path = os.path.join(output_dir, filename_summary)

        final_df.to_csv(full_path, index=False, encoding="utf-8-sig")
        print(f"✅ 第 {i+1} 组分析完成，已保存汇总表：{full_path}")

    else:
        print(f"第 {i+1} 组分析 (D={D}, T={T}) 无有效数据，未生成报表。")

# ========== 批量分析完成 ==========
print(f"\n{'='*60}")
print("所有分析已完成！")
print(f"{'='*60}")
