import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, auc

# 调整整导入，使用eva.py中的load_model替代detect_checkpoint_and_load_model
from model import BindingSitePredictor
from eva import load_model  # 改用eva.py中的模型加载函数
from dataloader import get_binding_site_dataloader
from train import TRAIN_CONFIG


def find_optimal_threshold():
    """寻找使模型指标（默认F1）最优的结合位点预测阈值"""
    print("=== 寻找最佳结合位点预测阈值 ===")
    print(f"📌 基础配置：")
    print(f"   - 搜索范围：0.1到0.9（步长0.01）")
    print(f"   - 优化目标：F1分数最大值")

    # 1. 加载最优模型（改用eva.py的load_model，避免KeyError）
    model, model_path, model_type = load_model()
    if model_type == "random":
        print("❌ 未找到训练好的模型，使用随机初始化模型无意义")
        return

    model.eval()
    print(f"✅ 模型加载完成：{model_path}")

    # 2. 加载测试集数据
    test_loader = get_binding_site_dataloader(split="test")
    print(f"✅ 测试集加载完成：{len(test_loader.dataset)}个样本")

    # 3. 获取所有预测概率和真实标签
    all_preds = []
    all_targets = []

    with torch.no_grad(), tqdm(
            test_loader,
            total=len(test_loader),
            desc="获取预测结果"
    ) as pbar:
        for batch in pbar:
            _, pred_list = model(batch)
            all_preds.extend([p.squeeze(-1).cpu().numpy() for p in pred_list])
            all_targets.extend([t.cpu().numpy() for t in batch["binding_site_label"]])

    # 展平所有预测和标签（适配变长序列）
    pred_flat = []
    target_flat = []
    for pred, target in zip(all_preds, all_targets):
        valid_mask = np.isfinite(pred) & np.isfinite(target)
        pred_flat.extend(pred[valid_mask])
        target_flat.extend(target[valid_mask])

    pred_flat = np.array(pred_flat)
    target_flat = np.array(target_flat).astype(int)

    # 4. 搜索最佳阈值
    thresholds = np.arange(0.1, 0.97, 0.01)  # 从0.1到0.9，步长0.01
    best_f1 = -1.0
    best_threshold = 0.5  # 默认阈值
    f1_scores = []

    print("\n📊 开始阈值搜索...")
    for threshold in tqdm(thresholds, desc="测试阈值"):
        pred_binary = (pred_flat >= threshold).astype(int)
        f1 = f1_score(target_flat, pred_binary, zero_division=0)
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 5. 计算最佳阈值对应的完整指标
    best_pred_binary = (pred_flat >= best_threshold).astype(int)
    TP = np.sum((best_pred_binary == 1) & (target_flat == 1))
    TN = np.sum((best_pred_binary == 0) & (target_flat == 0))
    FP = np.sum((best_pred_binary == 1) & (target_flat == 0))
    FN = np.sum((best_pred_binary == 0) & (target_flat == 1))

    # 计算辅助指标
    sen = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # 6. 输出结果
    print("\n=== 最佳阈值搜索结果 ===")
    print(f"🏆 最佳阈值：{best_threshold:.2f}")
    print(f"   - 对应F1分数：{best_f1:.4f}")
    print(f"   - 灵敏度(Sen)：{sen:.4f}")
    print(f"   - 特异度(Spe)：{spe:.4f}")
    print(f"   - 精确率(Pre)：{pre:.4f}")

    # 7. 保存结果
    result_dir = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "threshold_analysis")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(
        result_dir,
        f"threshold_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # 保存所有阈值的F1分数
    pd.DataFrame({
        "threshold": thresholds,
        "f1_score": f1_scores
    }).to_csv(result_path, index=False)
    print(f"\n✅ 阈值分析结果已保存至：{result_path}")

    return best_threshold, best_f1


if __name__ == "__main__":
    find_optimal_threshold()