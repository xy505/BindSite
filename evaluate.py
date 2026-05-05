import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc  # 保留必要导入

import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 导入核心依赖
# --------------------------
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
from train import (
    TRAIN_CONFIG,
    set_ablation_mode
)


# --------------------------
# 2. 简化：仅保留AUC计算逻辑（修复单类别报错）
# --------------------------
def calculate_single_protein_auc(true_label_valid, pred_prob_valid):
    """
    仅计算单个蛋白质的AUC-ROC指标，处理单类别情况
    """
    target_binary = true_label_valid.astype(int)
    pred_flat = pred_prob_valid

    # 修复单类别报错：添加异常捕获+明确的条件判断
    try:
        unique_classes = np.unique(target_binary)
        if len(unique_classes) == 2:
            auc_score = roc_auc_score(target_binary, pred_flat) * 100
        else:
            # 仅一类标签时设为50（随机水平），并打印提示
            auc_score = 50.0
            print(f"\n⚠️  该蛋白质仅包含{unique_classes[0]}类标签，AUC-ROC设为随机水平50.0%")
    except Exception as e:
        # 捕获所有异常，确保程序不崩溃
        print(f"\n⚠️  计算AUC时出错：{str(e)}，已设为50.0%")
        auc_score = 50.0

    return round(auc_score, 2)


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
# 4. 测试集评估函数（仅输出AUC指标）
# --------------------------
def evaluate_test_set(ablation_type=None):
    print("=" * 80)
    print("=== 测试集评估（仅分析第一个蛋白质的AUC指标） ===")
    print(f"📌 基础配置：")
    print(f"   - 硬件设备：{MODEL_CONFIG['device']}")
    print(f"   - 测试集路径：{DATALOADER_CONFIG['pdb_dir']}")
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
        print(f"   - 测试集规模：{total_proteins} 个蛋白质")
        print(f"   - 批次数：{total_batches} 批次")
    except Exception as e:
        raise RuntimeError(f"❌ 测试集加载失败：{str(e)}\n")

    # 步骤3：仅处理第一个批次的第一个蛋白质
    print(f"\n📊 开始分析第一个蛋白质...")
    model.eval()
    first_protein_auc = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > 0:
                break

            loss, pred_list = model(batch)

            # 提取第一个蛋白质的数据
            first_true_label = batch["binding_site_label"][0].cpu().numpy()
            first_pred_prob = pred_list[0].squeeze(-1).cpu().detach().numpy()

            # 过滤有效标签（排除无穷值）
            valid_mask = np.isfinite(first_pred_prob) & np.isfinite(first_true_label)
            first_true_label_valid = first_true_label[valid_mask]
            first_pred_prob_valid = first_pred_prob[valid_mask]

            # 仅计算并输出AUC指标
            print("\n" + "=" * 80)
            print("=== 第一个蛋白质核心指标 ===")
            first_protein_auc = calculate_single_protein_auc(first_true_label_valid, first_pred_prob_valid)
            print(f"第一个蛋白质的AUC-ROC指标：{first_protein_auc}%")

            # 补充基础信息（可选，帮助理解AUC结果）
            print(f"\n基础信息参考：")
            print(f"   - 有效残基数：{len(first_true_label_valid)}")
            print(f"   - 真实结合位点数量：{np.sum(first_true_label_valid == 1)} 个")
            print(f"   - 预测阈值：{TRAIN_CONFIG['threshold']}")

            break  # 处理完第一个批次后立即终止

    if first_protein_auc is None:
        print("❌ 未计算出AUC指标")
    else:
        print(f"\n✅ 分析完成！第一个蛋白质的AUC-ROC：{first_protein_auc}%")

    return first_protein_auc, None


# --------------------------
# 5. 执行评估（脚本入口）
# --------------------------
if __name__ == "__main__":
    try:
        evaluate_test_set(ablation_type=None)
    except Exception as e:
        print(f"\n❌ 分析失败：{str(e)}")
        print(f"   排查步骤：")
        print(f"   1. 确认dataloader.py中PDB路径正确；")
        print(f"   2. 确认model.py的forward输出格式正确；")
        print(f"   3. 检查接触矩阵生成函数是否正常；")
        print(f"   4. Windows系统请将DataLoader的num_workers设为0。")