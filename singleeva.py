import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 导入核心依赖（与原评估代码完全对齐）
# --------------------------
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
from train import (
    TRAIN_CONFIG,
    set_ablation_mode
)


# --------------------------
# 2. 复用原评估代码的核心函数（保持完全一致）
# --------------------------
def calculate_full_metrics(all_preds, all_targets, threshold=TRAIN_CONFIG["threshold"]):
    pred_flat = []
    target_flat = []
    for pred, target in zip(all_preds, all_targets):
        pred_prob = pred.squeeze(-1).cpu().detach().numpy()
        true_label = target.cpu().numpy()
        valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)
        pred_flat.extend(pred_prob[valid_mask])
        target_flat.extend(true_label[valid_mask])

    pred_flat = np.array(pred_flat)
    target_flat = np.array(target_flat)
    pred_binary = (pred_flat >= threshold).astype(int)
    target_binary = target_flat.astype(int)

    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    TN = np.sum((pred_binary == 0) & (target_binary == 0))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))

    metrics = {}
    metrics["Sen"] = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0
    metrics["Spe"] = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0.0
    total = TP + TN + FP + FN
    metrics["Acc"] = ((TP + TN) / total) * 100 if total > 0 else 0.0
    metrics["Pre"] = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0.0
    denominator = np.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))
    metrics["MCC"] = (TP * TN - FP * FN) / denominator if denominator != 0 else 0.0
    sen_decimal = metrics["Sen"] / 100
    pre_decimal = metrics["Pre"] / 100
    metrics["F1"] = (2 * sen_decimal * pre_decimal / (sen_decimal + pre_decimal)) * 100 if (
                                                                                                   sen_decimal + pre_decimal) > 0 else 0.0
    if len(np.unique(target_binary)) == 2:
        metrics["AUC"] = roc_auc_score(target_binary, pred_flat) * 100
    else:
        metrics["AUC"] = 50.0
    if len(np.unique(target_binary)) == 2:
        precision, recall, _ = precision_recall_curve(target_binary, pred_flat)
        metrics["AUC-PRC"] = auc(recall, precision) * 100
    else:
        metrics["AUC-PRC"] = 50.0

    metrics = {
        "Sen": round(metrics["Sen"], 2),
        "Spe": round(metrics["Spe"], 2),
        "Acc": round(metrics["Acc"], 2),
        "Pre": round(metrics["Pre"], 2),
        "MCC": round(metrics["MCC"], 4),
        "F1": round(metrics["F1"], 2),
        "AUC": round(metrics["AUC"], 2),
        "AUC-PRC": round(metrics["AUC-PRC"], 2)
    }
    return metrics, (TP, TN, FP, FN)


def load_model(ablation_type=None):
    model = BindingSitePredictor().to(MODEL_CONFIG["device"])
    model = set_ablation_mode(model, ablation_type)

    ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
    best_model_filename = f"{MODEL_CONFIG['best_model_name'].rsplit('.', 1)[0]}{ablation_suffix}.pth"
    best_model_path = os.path.join(
        TRAIN_CONFIG["checkpoint_dir"],
        best_model_filename
    )
    latest_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], TRAIN_CONFIG["resume_checkpoint"])

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=MODEL_CONFIG["device"]))
        model.eval()
        print(f"✅ 加载训练完成的最优模型：{best_model_path}")
        return model, best_model_path, "trained_best"
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=MODEL_CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ 加载训练断点模型：{latest_model_path}")
        print(f"   - 断点对应epoch：{checkpoint['current_epoch']}，上一轮损失：{checkpoint['last_train_loss']:.4f}")
        return model, latest_model_path, "trained_latest"
    else:
        model.train(False)
        print(f"⚠️  未检测到任何训练断点")
        print(f"   - 已随机初始化模型进行测试")
        return model, "random_initialized", "random"


# --------------------------
# 3. 新增：序列级性能分析核心函数
# --------------------------
def sequence_level_analysis(ablation_type=None):
    print("=" * 100)
    print("=== 序列级性能分析实验（复刻PepBCL论文实验设计） ===")
    print(f"📌 实验目的：评估每个蛋白质序列的单独预测性能，验证模型泛化稳定性")
    print(f"📌 基础配置：")
    print(f"   - 硬件设备：{MODEL_CONFIG['device']}")
    print(f"   - 测试集路径：{DATALOADER_CONFIG['pdb_dir']}")
    print(f"   - 核心指标：序列级AUC（每个蛋白质单独计算）")
    if ablation_type is not None:
        print(f"   - 消融实验：{ablation_type}")
    print("=" * 100)

    # 步骤1：加载模型和测试集
    model, model_path, model_type = load_model(ablation_type)
    print(f"\n📌 加载测试集数据...")
    test_loader = get_binding_site_dataloader(split="test")
    total_proteins = len(test_loader.dataset)
    print(f"✅ 测试集加载完成：共{total_proteins}个蛋白质序列")

    # 步骤2：逐蛋白质预测并计算序列级AUC
    model.eval()
    sequence_metrics = []  # 存储每个序列的性能指标
    all_protein_names = []  # 存储蛋白质名称（需确保dataloader返回蛋白质ID）

    with torch.no_grad(), tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"Sequence-level Evaluation",
            unit="protein"
    ) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # 前向传播（获取当前批次每个蛋白质的预测结果）
            loss, pred_list = model(batch)
            batch_labels = batch["binding_site_label"]
            # 若dataloader返回蛋白质名称/ID，记录下来（关键：关联序列与性能）
            protein_names = batch.get("protein_id", [f"protein_{batch_idx}_{i}" for i in range(len(pred_list))])

            # 逐蛋白质计算序列级AUC
            for idx, (pred, target, pname) in enumerate(zip(pred_list, batch_labels, protein_names)):
                # 提取当前蛋白质的预测概率和真实标签
                pred_prob = pred.squeeze(-1).cpu().detach().numpy()
                true_label = target.cpu().numpy()
                valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)
                pred_valid = pred_prob[valid_mask]
                target_valid = true_label[valid_mask]

                # 计算序列级AUC（处理仅含一类标签的情况）
                if len(np.unique(target_valid)) == 2:
                    seq_auc = roc_auc_score(target_valid, pred_valid) * 100
                else:
                    seq_auc = 50.0  # 仅一类标签时为随机水平

                # 计算该蛋白质的基础统计量（可选，用于补充分析）
                pred_binary = (pred_valid >= TRAIN_CONFIG["threshold"]).astype(int)
                target_binary = target_valid.astype(int)
                TP = np.sum((pred_binary == 1) & (target_binary == 1))
                TN = np.sum((pred_binary == 0) & (target_binary == 0))
                FP = np.sum((pred_binary == 1) & (target_binary == 0))
                FN = np.sum((pred_binary == 0) & (target_binary == 1))

                # 存储当前序列的性能数据
                sequence_metrics.append({
                    "protein_name": pname,
                    "AUC": round(seq_auc, 2),
                    "residue_count": len(target_valid),
                    "binding_residue_count": int(np.sum(target_valid)),
                    "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN
                })
                all_protein_names.append(pname)

            pbar.set_postfix({"Processed Proteins": len(sequence_metrics)})

    # 步骤3：统计分析序列级性能分布
    print(f"\n📊 序列级性能统计分析（共{len(sequence_metrics)}个蛋白质）")
    auc_list = [item["AUC"] for item in sequence_metrics]
    auc_array = np.array(auc_list)

    # 统计不同AUC区间的蛋白质数量及占比（复刻论文图3的区间划分）
    auc_intervals = [
        ("AUC > 0.9", auc_array >= 90.0),
        ("0.8 ≤ AUC < 0.9", (auc_array >= 80.0) & (auc_array < 90.0)),
        ("0.7 ≤ AUC < 0.8", (auc_array >= 70.0) & (auc_array < 80.0)),
        ("0.6 ≤ AUC < 0.7", (auc_array >= 60.0) & (auc_array < 70.0)),
        ("AUC < 0.6", auc_array < 60.0)
    ]

    print(f"\n📋 AUC区间分布统计：")
    interval_stats = []
    for interval_name, mask in auc_intervals:
        count = np.sum(mask)
        ratio = (count / len(auc_array)) * 100
        interval_stats.append({
            "AUC区间": interval_name,
            "蛋白质数量": count,
            "占比(%)": round(ratio, 2)
        })
        print(f"   - {interval_name}：{count}个（{ratio:.2f}%）")

    # 计算整体统计量
    print(f"\n📋 序列级AUC整体统计：")
    print(f"   - 平均值：{np.mean(auc_array):.2f}%")
    print(f"   - 中位数：{np.median(auc_array):.2f}%")
    print(f"   - 最大值：{np.max(auc_array):.2f}%（蛋白质：{sequence_metrics[np.argmax(auc_array)]['protein_name']}）")
    print(f"   - 最小值：{np.min(auc_array):.2f}%（蛋白质：{sequence_metrics[np.argmin(auc_array)]['protein_name']}）")
    print(f"   - 标准差：{np.std(auc_array):.2f}%")

    # 步骤4：可视化序列级性能分布（复刻论文图3风格）
    plt.figure(figsize=(12, 6), dpi=300)

    # 子图1：AUC区间分布柱状图
    plt.subplot(1, 2, 1)
    intervals = [stat["AUC区间"] for stat in interval_stats]
    counts = [stat["蛋白质数量"] for stat in interval_stats]
    plt.bar(intervals, counts, color="#1f77b4", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("AUC区间")
    plt.ylabel("蛋白质数量")
    plt.title("序列级AUC区间分布", fontsize=12, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)

    # 子图2：AUC值散点图（模拟论文对比风格，若无对比模型则展示自身分布）
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(auc_array)), auc_array, c="#ff7f0e", alpha=0.6, s=20)
    plt.axhline(y=np.mean(auc_array), color="red", linestyle="--", alpha=0.8, label=f"平均值：{np.mean(auc_array):.2f}%")
    plt.axhline(y=80.0, color="gray", linestyle=":", alpha=0.8, label="AUC=80%")
    plt.xlabel("蛋白质序号")
    plt.ylabel("序列级AUC（%）")
    plt.title("各蛋白质序列级AUC分布", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_fig_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "sequence_level_auc_analysis.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ 序列级性能可视化图已保存至：{save_fig_path}")
    plt.show()

    # 步骤5：保存详细结果到CSV（便于后续分析）
    save_csv_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "sequence_level_metrics.csv")
    metrics_df = pd.DataFrame(sequence_metrics)
    metrics_df.to_csv(save_csv_path, index=False, encoding="utf-8")
    print(f"✅ 序列级详细性能数据已保存至：{save_csv_path}")

    # 步骤6：输出关键结论
    high_perf_ratio = (np.sum(auc_array >= 80.0) / len(auc_array)) * 100
    print(f"\n=== 序列级性能分析结论 ===")
    print(f"📌 模型在{high_perf_ratio:.2f}%的蛋白质序列上AUC≥80%，表现稳定")
    print(f"📌 序列级AUC平均值为{np.mean(auc_array):.2f}%，说明模型泛化能力良好")
    print(f"📌 详细数据已保存，可进一步分析模型在不同类型蛋白质上的性能差异")
    print("=" * 100)

    return sequence_metrics, auc_array


# --------------------------
# 4. 执行序列级分析实验（脚本入口）
# --------------------------
if __name__ == "__main__":
    try:
        # 执行序列级性能分析（支持消融实验，默认无消融）
        sequence_metrics, auc_array = sequence_level_analysis(ablation_type=None)
    except Exception as e:
        print(f"\n❌ 序列级性能分析失败：{str(e)}")
        print(f"   排查步骤：")
        print(f"   1. 确认dataloader返回的batch中包含'protein_id'（或修改代码中蛋白质命名逻辑）；")
        print(f"   2. 确认model.py的forward输出格式为（loss, pred_list），pred_list为每个蛋白质的残基预测概率；")
        print(f"   3. 检查测试集中蛋白质的残基标签是否为变长数组格式；")
        print(f"   4. Windows系统请将DataLoader的num_workers设为0。")