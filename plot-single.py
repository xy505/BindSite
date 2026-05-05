import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 配置参数（修改为你的保存文件路径）
# --------------------------
TRAIN_CONFIG = {
    "checkpoint_dir": "./bs_checkpoints"  # 与代码1的保存目录一致
}
CURRENT_MODEL_CSV = "current_model_sequence_auc.csv"  # 代码1中当前模型的保存文件名
REFERENCE_MODEL_CSV = "reference_model_sequence_auc.csv"  # 代码1中参考模型的保存文件名

# 新增：统一控制蛋白质ID字体大小（修改这里即可全局生效）
PROTEIN_ID_FONTSIZE = 11  # 原数值是9，可根据需要调整（如8/10/12/14）
# 新增：AUC过滤阈值
AUC_THRESHOLD = 50.0  # 过滤掉AUC低于该值的蛋白质

# 新增：标注文字偏移量配置（控制ID离节点的距离，可按需调整）
RED_LABEL_OFFSETS = [(-13, 13), (13, 13), (-13, 13), (13, 13)]  # 红色节点ID偏移（左/右上，距离更大）
GREEN_LEFT_OFFSET = (-10, 10)  # 绿色左节点ID偏移（左上）
GREEN_RIGHT_OFFSET = (10, 10)  # 绿色右节点ID偏移（右上）


# --------------------------
# 2. 核心函数：加载、过滤并匹配两个模型的结果
# --------------------------
def load_filter_and_match_auc_results(current_csv, reference_csv):
    """
    加载两个模型的CSV结果，过滤AUC<50的记录，按蛋白质名称匹配（以数量少的为基准）
    """
    # 加载CSV文件
    current_df = pd.read_csv(os.path.join(TRAIN_CONFIG["checkpoint_dir"], current_csv))
    reference_df = pd.read_csv(os.path.join(TRAIN_CONFIG["checkpoint_dir"], reference_csv))

    # 验证必要列存在
    for df, name in [(current_df, "Current"), (reference_df, "Reference")]:
        if "Protein_Name" not in df.columns or "AUC(%)" not in df.columns:
            raise ValueError(f"{name}模型CSV文件缺少必要列：需要包含'Protein_Name'和'AUC(%)'")

    # --------------------------
    # 第一步：过滤AUC低于阈值的记录
    # --------------------------
    current_filtered = current_df[current_df["AUC(%)"] >= AUC_THRESHOLD].copy()
    reference_filtered = reference_df[reference_df["AUC(%)"] >= AUC_THRESHOLD].copy()

    print(f"🔍 AUC过滤结果（阈值={AUC_THRESHOLD}%）:")
    print(f"   - Current model: 原始{len(current_df)}条 → 过滤后{len(current_filtered)}条")
    print(f"   - Reference model: 原始{len(reference_df)}条 → 过滤后{len(reference_filtered)}条")

    # --------------------------
    # 第二步：按蛋白质名称匹配（以数量少的为基准）
    # --------------------------
    # 获取两个过滤后数据集的蛋白质名称集合
    current_proteins = set(current_filtered["Protein_Name"].unique())
    reference_proteins = set(reference_filtered["Protein_Name"].unique())

    # 确定基准数据集（数量少的）
    if len(current_proteins) <= len(reference_proteins):
        base_proteins = current_proteins
        base_name = "Current"
        other_name = "Reference"
    else:
        base_proteins = reference_proteins
        base_name = "Reference"
        other_name = "Current"

    print(f"\n📊 匹配基准选择:")
    print(
        f"   - {base_name}模型蛋白质数量({len(base_proteins)}) ≤ {other_name}模型({len(reference_proteins if base_name == 'Current' else current_proteins)})")
    print(f"   - 以{base_name}模型为基准进行匹配")

    # 只保留基准数据集中存在的蛋白质
    current_filtered_matched = current_filtered[current_filtered["Protein_Name"].isin(base_proteins)].copy()
    reference_filtered_matched = reference_filtered[reference_filtered["Protein_Name"].isin(base_proteins)].copy()

    # 按蛋白质名称合并（确保1:1匹配）
    merged_df = pd.merge(
        current_filtered_matched.rename(columns={"AUC(%)": "Current_AUC(%)"}),
        reference_filtered_matched.rename(columns={"AUC(%)": "Reference_AUC(%)"}),
        on="Protein_Name",
        how="inner"  # 内连接确保只保留双方都存在的
    )

    # 去重（确保每个蛋白质只保留一条记录）
    merged_df = merged_df.drop_duplicates(subset=["Protein_Name"], keep="first")

    print(f"\n✅ 最终匹配结果:")
    print(f"   - 匹配的蛋白质数量: {len(merged_df)}")
    print(f"   - Current model 平均AUC: {merged_df['Current_AUC(%)'].mean():.2f}%")
    print(f"   - Reference model 平均AUC: {merged_df['Reference_AUC(%)'].mean():.2f}%")

    return merged_df


# --------------------------
# 3. 绘制论文图3风格的对比图（无直线，ID远离节点）
# --------------------------
def plot_paper_figure3(merged_df):
    """
    复刻论文图3风格：散点图 + 内嵌统计表格 + 差异化标注
    - 移除所有直线引线
    - 蛋白质ID标注远离节点（通过大偏移量实现）
    - 红色节点：4个提升显著的蛋白质，ID交替左上/右上
    - 绿色节点：2个最优表现蛋白质，ID按x轴位置左上/右上
    """
    # 复制数据避免修改原数据
    df = merged_df.copy()

    # 计算AUC提升值
    df["AUC_Improvement"] = df["Current_AUC(%)"] - df["Reference_AUC(%)"]

    current_auc = df["Current_AUC(%)"].values
    reference_auc = df["Reference_AUC(%)"].values
    protein_names = df["Protein_Name"].values
    improvement = df["AUC_Improvement"].values

    # 统计不同AUC区间分布
    def calculate_auc_intervals(auc_array):
        intervals = [
            ("AUC > 0.9", auc_array >= 90.0),
            ("0.8 ≤ AUC < 0.9", (auc_array >= 80.0) & (auc_array < 90.0)),
            ("0.7 ≤ AUC < 0.8", (auc_array >= 70.0) & (auc_array < 80.0)),
            ("0.6 ≤ AUC < 0.7", (auc_array >= 60.0) & (auc_array < 70.0)),
            ("AUC < 0.6", auc_array < 60.0)
        ]
        stats = []
        for name, mask in intervals:
            ratio = (np.sum(mask) / len(auc_array)) * 100 if len(auc_array) > 0 else 0.0
            stats.append(round(ratio, 1))
        return stats

    current_intervals = calculate_auc_intervals(current_auc)
    reference_intervals = calculate_auc_intervals(reference_auc)
    interval_names = ["AUC > 0.9", "0.8 ≤ AUC < 0.9", "0.7 ≤ AUC < 0.8", "0.6 ≤ AUC < 0.7", "AUC < 0.6"]

    # 空数据检查
    if len(df) == 0:
        print("❌ 没有可用数据进行绘图（过滤后无匹配的蛋白质）")
        return

    # 创建图表
    plt.figure(figsize=(10, 8), dpi=300)

    # 绘制散点图（基础散点）
    scatter = plt.scatter(
        reference_auc, current_auc,
        c="#ff7f0e", alpha=0.6, s=30, edgecolors="black", linewidth=0.5,
        label="All Proteins"
    )

    # 绘制对角线（性能相等线）
    plt.plot([AUC_THRESHOLD, 105], [AUC_THRESHOLD, 105], "k--", alpha=0.8, label="Performance equality line")

    # --------------------------
    # 标注类别1：4个提升显著的蛋白质（红色节点，无直线，ID远离节点）
    # --------------------------
    if len(df) >= 4:
        significant_improve_mask = (reference_auc < 70) & (current_auc > 80) & (improvement > 15)
        significant_proteins = df[significant_improve_mask].copy()

        # 兜底策略，确保选够4个
        if len(significant_proteins) < 4:
            significant_proteins = df.sort_values("AUC_Improvement", ascending=False).head(4)
        else:
            significant_proteins = significant_proteins.head(4)

        # 绘制红色节点 + 远离的ID标注（无直线）
        for idx, (row, offset) in enumerate(zip(significant_proteins.iterrows(), RED_LABEL_OFFSETS)):
            row = row[1]
            # 绘制红色标注点
            plt.scatter(
                row["Reference_AUC(%)"], row["Current_AUC(%)"],
                c="red", s=30, edgecolors="black", linewidth=0.5, alpha=0.8,
            )
            # 添加蛋白质ID标注（无直线，大偏移量）
            plt.annotate(
                row["Protein_Name"],
                xy=(row["Reference_AUC(%)"], row["Current_AUC(%)"]),
                xytext=offset,  # 大偏移量，ID远离节点
                textcoords="offset points",
                fontsize=9,  # 增大字体，无直线时更清晰
                color="darkred",
                fontweight="bold",
                alpha=0.95,
                ha="center", va="center",  # 文字居中
                # 可选：添加轻微白色背景，避免被散点遮挡
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )
    else:
        print("⚠️ 数据量不足（<4条），跳过显著提升蛋白质标注")

    # --------------------------
    # 标注类别2：2个最优表现的蛋白质（绿色节点，无直线，ID远离节点）
    # --------------------------
    if len(df) >= 2:
        # 筛选出当前模型AUC最高的2个蛋白质
        top_performance_proteins = df.sort_values("Current_AUC(%)", ascending=False).head(2).copy()
        top_performance_proteins = top_performance_proteins.sort_values("Reference_AUC(%)")
        left_protein = top_performance_proteins.iloc[0]
        right_protein = top_performance_proteins.iloc[1]

        # 绘制左绿色节点 + ID（左上偏移）
        plt.scatter(
            left_protein["Reference_AUC(%)"], left_protein["Current_AUC(%)"],
            c="darkgreen", s=30, edgecolors="black", linewidth=0.5, alpha=0.8,
        )
        plt.annotate(
            left_protein["Protein_Name"],
            xy=(left_protein["Reference_AUC(%)"], left_protein["Current_AUC(%)"]),
            xytext=GREEN_LEFT_OFFSET,  # 左上大偏移
            textcoords="offset points",
            fontsize=9,
            color="darkgreen",
            fontweight="bold",
            alpha=0.95,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

        # 绘制右绿色节点 + ID（右上偏移）
        plt.scatter(
            right_protein["Reference_AUC(%)"], right_protein["Current_AUC(%)"],
            c="darkgreen", s=30, edgecolors="black", linewidth=0.5, alpha=0.8,
        )
        plt.annotate(
            right_protein["Protein_Name"],
            xy=(right_protein["Reference_AUC(%)"], right_protein["Current_AUC(%)"]),
            xytext=GREEN_RIGHT_OFFSET,  # 右上大偏移
            textcoords="offset points",
            fontsize=9,
            color="darkgreen",
            fontweight="bold",
            alpha=0.95,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

        # 输出节点位置信息
        print(
            f"🟢 {left_protein['Protein_Name']}: x轴位置={left_protein['Reference_AUC(%)']:.2f}, 标注位置=左上（偏移{GREEN_LEFT_OFFSET}）")
        print(
            f"🟢 {right_protein['Protein_Name']}: x轴位置={right_protein['Reference_AUC(%)']:.2f}, 标注位置=右上（偏移{GREEN_RIGHT_OFFSET}）")
    else:
        print("⚠️ 数据量不足（<2条），跳过最优表现蛋白质标注")

    # --------------------------
    # 图表最终配置 - 表格修改核心区【表头浅灰色背景】
    # --------------------------
    # 内嵌统计表格
    table_data = []
    for i in range(len(interval_names)):
        table_data.append([
            interval_names[i],
            f"{current_intervals[i]}%",
            f"{reference_intervals[i]}%"
        ])
    table = plt.table(
        cellText=table_data,
        colLabels=["AUC Interval", "DMGNN", "PDNAPred"],
        cellLoc="center",
        loc="lower right",
        bbox=[0.58, 0.05, 0.38, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # ============ 新增核心修改：设置表格第一行(表头)背景为浅灰色 ============
    for i in range(3):  # 表头共3列
        table[(0, i)].set_facecolor('#e6e6e6')  # 浅灰色背景色，可修改色值调整深浅
    # ====================================================================

    # 坐标轴与标题
    plt.xlabel("AUC (%) of PDNAPred", fontsize=12)
    plt.ylabel("AUC (%) of DMGNN", fontsize=12)
    plt.title(f"Sequence-level AUC Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.xlim(AUC_THRESHOLD, 105)
    plt.ylim(AUC_THRESHOLD, 105)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)

    # 保存图片
    save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"],
                             f"sequence_level_auc_comparison_no_line_auc{AUC_THRESHOLD}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✅ Figure saved to: {save_path}")

    # 输出关键统计信息
    better_count = np.sum(current_auc > reference_auc)
    better_ratio = (better_count / len(current_auc)) * 100 if len(current_auc) > 0 else 0.0
    print(f"\n=== Statistical Summary ===")
    print(f"📌 Current model outperforms reference model on {better_ratio:.1f}% of proteins")
    print(f"📌 Current model mean AUC: {current_auc.mean():.2f}%")
    print(f"📌 Reference model mean AUC: {reference_auc.mean():.2f}%")
    print(f"📌 AUC improvement (mean): {current_auc.mean() - reference_auc.mean():.2f}%")

    # 输出标注的蛋白质信息
    if len(df) >= 4:
        print(f"\n=== Annotated Proteins ===")
        print(f"🔴 Significant Improvement (4 proteins):")
        for _, row in significant_proteins.iterrows():
            print(
                f"   - {row['Protein_Name']}: Ref={row['Reference_AUC(%)']:.2f}%, Curr={row['Current_AUC(%)']:.2f}%, Δ={row['AUC_Improvement']:.2f}%")

    plt.show()


# --------------------------
# 4. 执行入口
# --------------------------
if __name__ == "__main__":
    try:
        # Step 1: 加载、过滤并匹配两个模型的结果
        merged_df = load_filter_and_match_auc_results(CURRENT_MODEL_CSV, REFERENCE_MODEL_CSV)

        if len(merged_df) == 0:
            print("\n❌ 错误：过滤和匹配后无可用数据，请检查：")
            print(f"   1. AUC阈值是否设置过高（当前={AUC_THRESHOLD}%）")
            print(f"   2. 两个CSV文件是否有共同的蛋白质名称")
            print(f"   3. CSV文件中的AUC值是否有效（数值型）")
        else:
            # Step 2: 绘制图表（无直线，ID远离节点）
            plot_paper_figure3(merged_df)

            # Step 3: 保存过滤匹配后的完整结果
            merged_save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"],
                                            f"merged_sequence_auc_comparison_auc{AUC_THRESHOLD}.csv")
            merged_df["AUC_Difference(%)"] = merged_df["Current_AUC(%)"] - merged_df["Reference_AUC(%)"]
            merged_df.to_csv(merged_save_path, index=False, encoding="utf-8")
            print(f"\n✅ Merged result saved to: {merged_save_path}")

    except Exception as e:
        print(f"\n❌ Execution failed: {str(e)}")
        print(f"   Troubleshooting Steps:")
        print(f"   1. Check if CSV files exist in checkpoint directory")
        print(f"   2. Ensure both CSV files have 'Protein_Name' and 'AUC(%)' columns")
        print(f"   3. Ensure AUC values are numeric (not string)")
        print(f"   4. Check if AUC threshold ({AUC_THRESHOLD}%) is reasonable")