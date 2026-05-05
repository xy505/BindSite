import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc  # 新增PR曲线相关导入

import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 导入核心依赖（与训练代码完全对齐，复用方案配置）
# --------------------------
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
from train import (
    TRAIN_CONFIG,
    set_ablation_mode
)


# --------------------------
# 2. 核心修改：扩展指标计算函数（新增AUC-PRC及Sen/Spe/Pre/MCC）
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
    return metrics, (TP, TN, FP, FN)


# --------------------------
# 3. 模型加载函数（保持不变）
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
# 4. 测试集全量评估函数（修改指标输出部分）
# --------------------------
def evaluate_test_set(ablation_type=None):
    print("=" * 80)
    print("=== 测试集评估（含AUC-PRC指标） ===")
    print(f"📌 基础配置：")
    print(f"   - 硬件设备：{MODEL_CONFIG['device']}")
    print(f"   - 测试集路径：{DATALOADER_CONFIG['pdb_dir']}")
    # print(f"   - 预测阈值：{TRAIN_CONFIG['threshold']}")
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
        # print(f"   - 测试集规模：{total_proteins} 个蛋白质")
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

    # 步骤4：计算测试集全量指标
    print(f"\n📌 计算测试集全量评估指标...")
    avg_test_loss = total_test_loss / total_proteins if total_proteins > 0 else 0.0
    test_metrics, (TP, TN, FP, FN) = calculate_full_metrics(all_preds, all_targets)
    # total_residues = len(all_targets)
    # binding_site_ratio = (TP + FN) / total_residues * 100 if total_residues > 0 else 0.0

    # 计算总残基数量（遍历每个样本的标签数组，累加长度）
    total_residues = sum(len(target) for target in all_targets)
    # 结合位点残基占比 = (真实结合位点残基总数 / 总残基数量) * 100%
    binding_site_ratio = (TP + FN) / total_residues * 100 if total_residues > 0 else 0.0

    # 步骤5：打印测试集评估结果（新增AUC-PRC输出）
    print(f"\n=== 测试集完整评估结果 ===")
    print(f"📋 一、基础统计信息：")
    print(f"   - 模型类型：{'训练完成的最优模型' if model_type == 'trained_best' else '训练断点模型' if model_type == 'trained_latest' else '随机初始化模型'}")
    print(f"   - 模型路径/状态：{model_path}")
    # print(f"   - 测试集规模：{total_proteins} 个蛋白质，{total_residues} 个残基")
    print(f"   - 结合位点残基占比：{binding_site_ratio:.2f}%")
    # print(f"   - 预测阈值：{TRAIN_CONFIG['threshold']}")
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
    # 新增AUC-PRC输出及解释
    print(f"   8. AUC-PRC：{test_metrics['AUC-PRC']:.2f}%")
    print(f"      - 含义：PR曲线下面积，更适合不平衡数据（结合位点占比低时参考价值更高）")

    print(f"\n📊 三、辅助指标：")
    print(f"   - 测试集全量平均损失：{avg_test_loss:.4f}")

    # 步骤6：保存测试集评估结果（包含AUC-PRC）
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
        f.write(f"   - 模型类型：{'训练完成的最优模型' if model_type == 'trained_best' else '训练断点模型' if model_type == 'trained_latest' else '随机初始化模型'}\n")
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