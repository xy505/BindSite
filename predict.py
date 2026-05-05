import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
# 导入原有模块（确保路径正确）
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG


# --------------------------
# 修复后的指标计算函数（包含正确的AUC计算）
# --------------------------
def calculate_metrics(pred_list, target_list):
    """修复维度挤压错误，正确计算AUC指标"""
    pred_flat = []
    target_flat = []
    for pred, target in zip(pred_list, target_list):
        # 处理预测结果：移除所有大小为1的维度
        pred_prob = np.squeeze(pred)
        # 确保是一维数组
        if pred_prob.ndim != 1:
            raise ValueError(f"预测结果维度错误，应为1维，实际为{pred_prob.ndim}维：{pred_prob.shape}")

        # 处理真实标签
        true_label = np.squeeze(target)
        if true_label.ndim != 1:
            raise ValueError(f"标签维度错误，应为1维，实际为{true_label.ndim}维：{true_label.shape}")

        # 过滤无效值
        valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)
        pred_flat.extend(pred_prob[valid_mask])
        target_flat.extend(true_label[valid_mask])

    # 转为numpy数组
    pred_flat = np.array(pred_flat)
    target_flat = np.array(target_flat)
    pred_binary = (pred_flat >= TEST_CONFIG["threshold"]).astype(int)
    target_binary = target_flat.astype(int)

    # 计算二分类统计量
    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    TN = np.sum((pred_binary == 0) & (target_binary == 0))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))

    # 计算指标
    metrics = {}
    metrics["sen"] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    metrics["spe"] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    total = TP + TN + FP + FN
    metrics["accuracy"] = (TP + TN) / total if total > 0 else 0.0
    metrics["pre"] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    denominator = np.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))
    metrics["mcc"] = (TP * TN - FP * FN) / denominator if denominator != 0 else 0.0
    metrics["f1"] = 2 * metrics["pre"] * metrics["sen"] / (metrics["pre"] + metrics["sen"]) if (metrics["pre"] +
                                                                                                metrics[
                                                                                                    "sen"]) > 0 else 0.0

    # 正确计算AUC（处理标签只有一类的情况）
    unique_labels = np.unique(target_binary)
    if len(unique_labels) < 2:
        # 若只有一类标签，AUC无意义，设为0.5（随机猜测水平）
        metrics["auc"] = 0.5
    else:
        # 使用sklearn的roc_auc_score计算
        metrics["auc"] = roc_auc_score(target_binary, pred_flat)

    return {k: round(v, 4) for k, v in metrics.items()}


# --------------------------
# 测试配置
# --------------------------
TEST_CONFIG = {
    "checkpoint_dir": "./bs_checkpoints",
    "best_model_name": "best_model.pth",
    "threshold": MODEL_CONFIG.get("threshold", 0.5),  # 预测阈值
    "ablation_type": "None",
    "device": MODEL_CONFIG["device"]
}

# 处理消融实验模型名
if TEST_CONFIG["ablation_type"] != "None":
    TEST_CONFIG["best_model_name"] = f"best_model_{TEST_CONFIG['ablation_type']}.pth"


def load_best_model():
    """加载最优模型"""
    model = BindingSitePredictor().to(TEST_CONFIG["device"])
    model_path = os.path.join(TEST_CONFIG["checkpoint_dir"], TEST_CONFIG["best_model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    model.load_state_dict(torch.load(model_path, map_location=TEST_CONFIG["device"]))
    model.eval()
    print(f"✅ 加载最优模型：{model_path}")
    return model


def test_all_samples(model):
    """预测测试集并收集结果（含前10个样本详情）"""
    test_loader = get_binding_site_dataloader(split="test")
    abnormal_samples = {"all_1": [], "all_0": []}
    all_preds = []  # 存储所有预测概率（numpy数组）
    all_targets = []  # 存储所有真实标签（numpy数组）
    top10_details = []  # 存储前10个样本的详细结果

    print(f"\n开始预测测试集（共{len(test_loader.dataset)}个样本）...")
    with torch.no_grad(), tqdm(test_loader, desc="测试进度", unit="批次") as pbar:
        for batch_idx, batch in enumerate(pbar):
            _, pred_list = model(batch)  # 模型预测（列表，每个元素为单个样本的预测）
            labels = batch["binding_site_label"]  # 真实标签（张量）

            # 逐个处理样本
            for sample_idx in range(len(pred_list)):
                # 预测概率：转为numpy并移除冗余维度
                pred_tensor = pred_list[sample_idx]  # 模型输出的张量
                pred_prob = pred_tensor.cpu().detach().numpy()  # 转为numpy
                pred_prob = np.squeeze(pred_prob)  # 移除所有size=1的维度

                # 真实标签：转为numpy并移除冗余维度
                true_label = labels[sample_idx].cpu().numpy()
                true_label = np.squeeze(true_label)

                # 二值化预测结果
                pred_binary = (pred_prob >= TEST_CONFIG["threshold"]).astype(int)

                # 检测全1/全0预测
                if np.all(pred_binary == 1):
                    abnormal_samples["all_1"].append((batch_idx, sample_idx))
                if np.all(pred_binary == 0):
                    abnormal_samples["all_0"].append((batch_idx, sample_idx))

                # 收集前10个样本的详情
                if len(top10_details) < 10:
                    top10_details.append({
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "pred_prob": pred_prob,
                        "pred_binary": pred_binary,
                        "true_label": true_label
                    })

                # 收集用于整体指标计算
                all_preds.append(pred_prob)
                all_targets.append(true_label)

    return abnormal_samples, all_preds, all_targets, top10_details


def print_results(abnormal_samples, all_preds, all_targets, top10_details):
    """打印结果（含前10个样本详情）"""
    print("\n" + "=" * 60)
    print("测试集结合位点预测结果汇总")
    print("=" * 60)

    # 1. 整体指标
    print("\n【1. 整体评估指标】")
    metrics = calculate_metrics(all_preds, all_targets)
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")

    # 2. 异常样本（全1/全0）
    print("\n【2. 异常预测样本】")
    print(f"  全1预测样本数：{len(abnormal_samples['all_1'])}")
    if abnormal_samples["all_1"]:
        print(f"  样本标识（批次, 样本索引）：{abnormal_samples['all_1'][:5]}...")  # 最多显示5个

    print(f"\n  全0预测样本数：{len(abnormal_samples['all_0'])}")
    if abnormal_samples["all_0"]:
        print(f"  样本标识（批次, 样本索引）：{abnormal_samples['all_0'][:5]}...")  # 最多显示5个

    # 3. 前10个样本的预测详情
    print("\n【3. 前10个样本预测详情】")
    for i, detail in enumerate(top10_details, 1):
        print(f"\n  样本{i}（批次{detail['batch_idx']}, 索引{detail['sample_idx']}）：")
        print(f"    序列长度：{len(detail['pred_binary'])}")
        print(f"    预测结合位点数（1的数量）：{np.sum(detail['pred_binary'])}")
        print(f"    真实结合位点数：{np.sum(detail['true_label'])}")
        # 打印前5个残基的预测概率和标签（避免过长）
        print("    前5个残基预测（概率 | 预测标签 | 真实标签）：")
        for j in range(min(5, len(detail['pred_binary']))):
            print(
                f"      残基{j}：{detail['pred_prob'][j]:.4f} | {detail['pred_binary'][j]} | {detail['true_label'][j]}")


if __name__ == "__main__":
    # 确保导入sklearn的roc_auc_score
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("缺少sklearn依赖，正在安装...")
        os.system("pip install scikit-learn")

    model = load_best_model()
    abnormal_samples, all_preds, all_targets, top10_details = test_all_samples(model)
    print_results(abnormal_samples, all_preds, all_targets, top10_details)