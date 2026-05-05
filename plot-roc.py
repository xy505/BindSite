import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import warnings

# 设置matplotlib参数，生成论文级别的图表
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
warnings.filterwarnings("ignore")

# --------------------------
# 原有依赖导入（保持不变）
# --------------------------
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
from train import (
    TRAIN_CONFIG,
    set_ablation_mode
)


# --------------------------
# 1. 原有指标计算函数（保持不变）
# --------------------------
def calculate_full_metrics(all_preds, all_targets, threshold=TRAIN_CONFIG["threshold"]):
    """
    计算完整评估指标（含方案要求的Sen/Spe/Acc/Pre/MCC/F1/AUC/AUC-PRC）：
    - 输入：全量预测概率列表、全量真实标签列表
    - 输出：完整指标字典（保留2-4位小数，百分比指标×100）
    """
    # 步骤1：展平残基级预测与标签（适配变长序列）
    pred_flat = []
    target_flat = []
    for pred, target in zip(all_preds, all_targets):
        pred_prob = pred.squeeze(-1).cpu().detach().numpy()
        true_label = target.cpu().numpy()
        valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)
        pred_flat.extend(pred_prob[valid_mask])
        target_flat.extend(true_label[valid_mask])

    # 转为numpy数组
    pred_flat = np.array(pred_flat)
    target_flat = np.array(target_flat)
    pred_binary = (pred_flat >= threshold).astype(int)
    target_binary = target_flat.astype(int)

    # 步骤2：计算二分类基础统计量
    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    TN = np.sum((pred_binary == 0) & (target_binary == 0))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))

    # 步骤3：计算方案要求的8类指标（新增AUC-PRC）
    metrics = {}
    # 1. 灵敏度（Sen）
    metrics["Sen"] = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0.0
    # 2. 特异度（Spe）
    metrics["Spe"] = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0.0
    # 3. 准确率（Acc）
    total = TP + TN + FP + FN
    metrics["Acc"] = ((TP + TN) / total) * 100 if total > 0 else 0.0
    # 4. 精确率（Pre）
    metrics["Pre"] = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0.0
    # 5. 马修斯相关系数（MCC）
    denominator = np.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))
    metrics["MCC"] = (TP * TN - FP * FN) / denominator if denominator != 0 else 0.0
    # 6. F1值
    sen_decimal = metrics["Sen"] / 100
    pre_decimal = metrics["Pre"] / 100
    metrics["F1"] = (2 * sen_decimal * pre_decimal / (sen_decimal + pre_decimal)) * 100 if (
                                                                                                   sen_decimal + pre_decimal) > 0 else 0.0
    # 7. AUC-ROC
    if len(np.unique(target_binary)) == 2:
        metrics["AUC"] = roc_auc_score(target_binary, pred_flat) * 100
    else:
        metrics["AUC"] = 50.0  # 仅一类标签时为随机水平
    # 新增：8. AUC-PRC（PR曲线下面积）
    if len(np.unique(target_binary)) == 2:
        # 计算PR曲线（precision-recall curve）
        precision, recall, _ = precision_recall_curve(target_binary, pred_flat)
        metrics["AUC-PRC"] = auc(recall, precision) * 100  # 转为百分比
    else:
        # 仅一类标签时，PR曲线无意义，设为0.5（随机水平）
        metrics["AUC-PRC"] = 50.0

    # 保留小数位数
    metrics = {
        "Sen": round(metrics["Sen"], 2),
        "Spe": round(metrics["Spe"], 2),
        "Acc": round(metrics["Acc"], 2),
        "Pre": round(metrics["Pre"], 2),
        "MCC": round(metrics["MCC"], 4),
        "F1": round(metrics["F1"], 2),
        "AUC": round(metrics["AUC"], 2),
        "AUC-PRC": round(metrics["AUC-PRC"], 2)  # 新增AUC-PRC的格式化
    }
    return metrics, (TP, TN, FP, FN), pred_flat, target_binary  # 新增返回展平后的预测值和标签


# --------------------------
# 2. 新增：绘制ROC和PR曲线的函数
# --------------------------
def plot_roc_pr_curves(pred_flat, target_flat, metrics, save_dir, ablation_type=None, model_type="trained_best"):
    """
    绘制并保存ROC曲线和PR曲线（合并在一张图或分开绘制）
    :param pred_flat: 展平后的预测概率数组
    :param target_flat: 展平后的真实标签数组
    :param metrics: 计算好的指标字典
    :param save_dir: 保存目录
    :param ablation_type: 消融实验类型（可选）
    :param model_type: 模型类型（trained_best/trained_latest/random）
    """
    # 创建保存子目录
    plot_dir = os.path.join(save_dir, "curves_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 生成文件名后缀
    ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
    time_suffix = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # 检查是否有两类标签（否则无法绘制有意义的曲线）
    if len(np.unique(target_flat)) != 2:
        print("⚠️  测试集仅包含一类标签，无法绘制有效的ROC/PR曲线")
        return

    # --------------------------
    # 绘制ROC曲线
    # --------------------------
    plt.figure(figsize=(6, 5))
    # 计算ROC曲线点
    fpr, tpr, _ = roc_curve(target_flat, pred_flat)
    roc_auc = metrics["AUC"] / 100  # 转回小数

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='#2E86AB', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    # 绘制随机基线
    plt.plot([0, 1], [0, 1], color='#A23B72', lw=1.5, linestyle='--',
             label='Random Classifier (AUC = 0.5)')

    # 图表美化
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve - Binding Site Prediction', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3, linestyle=':')
    plt.tight_layout()

    # 保存ROC曲线
    roc_save_path = os.path.join(
        plot_dir,
        f"roc_curve_{model_type}{ablation_suffix}_{time_suffix}.png"
    )
    plt.savefig(roc_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ ROC曲线已保存至：{roc_save_path}")

    # --------------------------
    # 绘制PR曲线
    # --------------------------
    plt.figure(figsize=(6, 5))
    # 计算PR曲线点
    precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
    pr_auc = metrics["AUC-PRC"] / 100  # 转回小数

    # 计算正样本比例（作为随机基线）
    pos_ratio = np.sum(target_flat) / len(target_flat)

    # 绘制PR曲线
    plt.plot(recall, precision, color='#F18F01', lw=2,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    # 绘制随机基线（正样本比例）
    plt.plot([0, 1], [pos_ratio, pos_ratio], color='#C73E1D', lw=1.5, linestyle='--',
             label=f'Random Classifier (Precision = {pos_ratio:.3f})')

    # 图表美化
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Binding Site Prediction', fontsize=14, pad=15)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3, linestyle=':')
    plt.tight_layout()

    # 保存PR曲线
    pr_save_path = os.path.join(
        plot_dir,
        f"pr_curve_{model_type}{ablation_suffix}_{time_suffix}.png"
    )
    plt.savefig(pr_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ PR曲线已保存至：{pr_save_path}")

    # --------------------------
    # 可选：绘制合并图（ROC+PR在一张图的两个子图）
    # --------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：ROC曲线
    ax1.plot(fpr, tpr, color='#2E86AB', lw=2,
             label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], color='#A23B72', lw=1.5, linestyle='--')
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3, linestyle=':')

    # 子图2：PR曲线
    ax2.plot(recall, precision, color='#F18F01', lw=2,
             label=f'AUC = {pr_auc:.3f}')
    ax2.plot([0, 1], [pos_ratio, pos_ratio], color='#C73E1D', lw=1.5, linestyle='--')
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3, linestyle=':')

    plt.tight_layout()
    combined_save_path = os.path.join(
        plot_dir,
        f"roc_pr_combined_{model_type}{ablation_suffix}_{time_suffix}.png"
    )
    plt.savefig(combined_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ ROC+PR合并曲线已保存至：{combined_save_path}")


# --------------------------
# 3. 原有模型加载函数（保持不变）
# --------------------------
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
# 4. 测试集全量评估函数（修改：调用绘图函数）
# --------------------------
def evaluate_test_set(ablation_type=None):
    print("=" * 80)
    print("=== 测试集评估（含AUC-PRC指标 + ROC/PR曲线绘制） ===")
    print(f"📌 基础配置：")
    print(f"   - 硬件设备：{MODEL_CONFIG['device']}")
    print(f"   - 测试集路径：{DATALOADER_CONFIG['pdb_dir']}")
    print(f"   - 评价指标：Sen/Spe/Acc/Pre/MCC/F1/AUC/AUC-PRC")
    if ablation_type is not None:
        print(f"   - 消融实验：{ablation_type}")
    print("=" * 80)

    # 步骤1：加载模型
    model, model_path, model_type = load_model(ablation_type)

    # 步骤2：加载测试集
    print(f"\n📌 加载测试集数据...")
    try:
        test_loader = get_binding_site_dataloader(split="test")
        total_proteins = len(test_loader.dataset)
        total_batches = len(test_loader)
        print(f"✅ 测试集加载完成：")
        print(f"   - 批次数：{total_batches} 批次")
    except Exception as e:
        raise RuntimeError(
            f"❌ 测试集加载失败：{str(e)}\n"
        )

    # 步骤3：测试集前向传播
    print(f"\n📊 开始测试集全量评估...")
    model.eval()
    total_test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad(), tqdm(
            test_loader,
            total=total_batches,
            desc=f"Evaluating Test Set",
            unit="batch",
            leave=True
    ) as pbar:
        for batch in pbar:
            loss, pred_list = model(batch)
            batch_size = len(batch["residue_sequence"])
            total_test_loss += loss.item() * batch_size
            all_preds.extend(pred_list)
            all_targets.extend(batch["binding_site_label"])
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    # 步骤4：计算测试集全量指标（新增返回pred_flat和target_flat）
    print(f"\n📌 计算测试集全量评估指标...")
    avg_test_loss = total_test_loss / total_proteins if total_proteins > 0 else 0.0
    test_metrics, (TP, TN, FP, FN), pred_flat, target_flat = calculate_full_metrics(all_preds, all_targets)

    # 计算总残基数量（遍历每个样本的标签数组，累加长度）
    total_residues = sum(len(target) for target in all_targets)
    # 结合位点残基占比 = (真实结合位点残基总数 / 总残基数量) * 100%
    binding_site_ratio = (TP + FN) / total_residues * 100 if total_residues > 0 else 0.0

    # 步骤5：打印测试集评估结果（保持不变）
    print(f"\n=== 测试集完整评估结果 ===")
    print(f"📋 一、基础统计信息：")
    print(
        f"   - 模型类型：{'训练完成的最优模型' if model_type == 'trained_best' else '训练断点模型' if model_type == 'trained_latest' else '随机初始化模型'}")
    print(f"   - 模型路径/状态：{model_path}")
    print(f"   - 结合位点残基占比：{binding_site_ratio:.2f}%")
    print(f"   - 二分类统计量：TP={TP}, TN={TN}, FP={FP}, FN={FN}")

    if model_type == "random":
        print(f"\n⚠️  重要提示：当前使用随机初始化模型，指标无实际生物学意义！")

    print(f"\n📋 二、完整评价指标（方案要求，基于全量残基计算）：")
    print(f"   1. 灵敏度（Sen）：{test_metrics['Sen']:.2f}%")
    print(f"      - 含义：真实结合位点残基中被正确预测的比例")
    print(f"   2. 特异度（Spe）：{test_metrics['Spe']:.2f}%")
    print(f"      - 含义：真实非结合位点残基中被正确预测的比例")
    print(f"   3. 准确率（Acc）：{test_metrics['Acc']:.2f}%")
    print(f"      - 含义：所有残基中预测正确的比例")
    print(f"   4. 精确率（Pre）：{test_metrics['Pre']:.2f}%")
    print(f"      - 含义：预测为结合位点的残基中实际正确的比例")
    print(f"   5. 马修斯相关系数（MCC）：{test_metrics['MCC']:.4f}")
    print(f"      - 含义：平衡类别不平衡的综合指标")
    print(f"   6. F1值：{test_metrics['F1']:.2f}%")
    print(f"      - 含义：Sen与Pre的调和平均")
    print(f"   7. AUC-ROC：{test_metrics['AUC']:.2f}%")
    print(f"      - 含义：模型区分结合位点与非结合位点的整体能力")
    print(f"   8. AUC-PRC：{test_metrics['AUC-PRC']:.2f}%")
    print(f"      - 含义：PR曲线下面积，更适合不平衡数据（结合位点占比低时参考价值更高）")

    print(f"\n📊 三、辅助指标：")
    print(f"   - 测试集全量平均损失：{avg_test_loss:.4f}")

    # 步骤6：保存测试集评估结果（保持不变）
    save_dir = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "test_results_full")
    os.makedirs(save_dir, exist_ok=True)
    model_type_suffix = "trained_best" if model_type == "trained_best" else "trained_latest" if model_type == "trained_latest" else "random_init"
    ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
    save_path = os.path.join(
        save_dir,
        f"test_full_metrics_{model_type_suffix}{ablation_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== 测试集完整评估结果 ===\n")
        f.write(f"评估时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"1. 模型信息：\n")
        f.write(
            f"   - 模型类型：{'训练完成的最优模型' if model_type == 'trained_best' else '训练断点模型' if model_type == 'trained_latest' else '随机初始化模型'}\n")
        f.write(f"   - 模型路径/状态：{model_path}\n")
        f.write(f"   - 消融模式：{ablation_type if ablation_type is not None else 'None'}\n")
        f.write(f"   - 预测阈值：{TRAIN_CONFIG['threshold']}\n")
        if model_type == "random":
            f.write(f"   - 风险提示：未训练模型，指标无实际意义\n")
        f.write(f"2. 测试集信息：\n")
        f.write(f"   - 蛋白质数量：{total_proteins} 个\n")
        f.write(f"   - 残基总数：{total_residues} 个\n")
        f.write(f"   - 结合位点残基占比：{binding_site_ratio:.2f}%\n")
        f.write(f"   - 二分类统计量：TP={TP}, TN={TN}, FP={FP}, FN={FN}\n")
        f.write(f"3. 辅助指标：\n")
        f.write(f"   - 测试集平均损失：{avg_test_loss:.4f}\n")
        f.write(f"4. 完整评价指标（%，除MCC外）：\n")
        for metric_name, metric_value in test_metrics.items():
            if metric_name != "MCC":
                f.write(f"   - {metric_name.upper()}：{metric_value:.2f}%\n")
            else:
                f.write(f"   - {metric_name.upper()}：{metric_value:.4f}\n")

    print(f"\n✅ 测试集完整评估结果已保存至：{save_path}")

    # 步骤7：新增：绘制并保存ROC/PR曲线
    print(f"\n📊 绘制ROC/PR曲线...")
    plot_roc_pr_curves(
        pred_flat=pred_flat,
        target_flat=target_flat,
        metrics=test_metrics,
        save_dir=save_dir,
        ablation_type=ablation_type,
        model_type=model_type
    )

    return test_metrics, avg_test_loss


# --------------------------
# 5. 执行评估（脚本入口）
# --------------------------
if __name__ == "__main__":
    try:
        evaluate_test_set(ablation_type=None)
    except Exception as e:
        print(f"\n❌ 测试集评估失败：{str(e)}")
        print(f"   排查步骤：")
        print(f"   1. 确认dataloader.py中PDB路径正确；")
        print(f"   2. 确认model.py的forward输出格式正确；")
        print(f"   3. 检查接触矩阵生成函数是否正常；")
        print(f"   4. Windows系统请将DataLoader的num_workers设为0。")