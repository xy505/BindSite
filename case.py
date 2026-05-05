import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import csv

import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 导入核心依赖（与原代码完全对齐）
# --------------------------
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
from train import (
    TRAIN_CONFIG,
    set_ablation_mode
)


# --------------------------
# 2. 模型加载函数（复用原逻辑）
# --------------------------
def load_model(ablation_type=None, custom_model_path=None):
    """加载模型（保持与原代码一致）"""
    model = BindingSitePredictor().to(MODEL_CONFIG["device"])
    model = set_ablation_mode(model, ablation_type)

    # 优先使用手动指定的模型路径
    if custom_model_path is not None and os.path.exists(custom_model_path):
        model.load_state_dict(torch.load(custom_model_path, map_location=MODEL_CONFIG["device"]))
        model.eval()
        print(f"✅ Loaded manually specified model: {custom_model_path}")
        return model
    # 原逻辑备用（可选）
    ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
    best_model_filename = f"{MODEL_CONFIG['best_model_name'].rsplit('.', 1)[0]}{ablation_suffix}.pth"
    best_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], best_model_filename)
    latest_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], TRAIN_CONFIG["resume_checkpoint"])

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=MODEL_CONFIG["device"]))
        model.eval()
        print(f"✅ Loaded best trained model: {best_model_path}")
        return model
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=MODEL_CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Loaded resume checkpoint model: {latest_model_path}")
        return model
    else:
        model.eval()
        print(f"⚠️ No trained checkpoint detected - using random model")
        return model


# --------------------------
# 3. 辅助函数：解析残基预测详情
# --------------------------
def analyze_residue_predictions(pred_prob, true_label, valid_mask, threshold=0.5):
    """
    分析单个蛋白质的残基预测详情
    :param pred_prob: 预测概率数组
    :param true_label: 真实标签数组
    :param valid_mask: 有效掩码
    :param threshold: 预测阈值（默认0.5）
    :return: 残基分析结果字典
    """
    # 过滤无效值
    pred_valid = pred_prob[valid_mask]
    target_valid = true_label[valid_mask]

    # 获取残基序号（注意：残基序号从1开始，而非0）
    residue_indices = np.where(valid_mask)[0]   # 转换为0-based序号

    # 预测标签（基于阈值）
    pred_label = (pred_valid >= threshold).astype(int)

    # 1. 真实结合位点序号（true=1）
    true_binding_residues = residue_indices[target_valid == 1].tolist()

    # 2. 正确预测的结合残基（true=1且pred=1）
    correct_binding = residue_indices[(target_valid == 1) & (pred_label == 1)].tolist()

    # 3. 误判为结合的非结合残基（false positive: true=0但pred=1）
    false_positive = residue_indices[(target_valid == 0) & (pred_label == 1)].tolist()

    # 4. 漏判的真实结合残基（false negative: true=1但pred=0）
    false_negative = residue_indices[(target_valid == 1) & (pred_label == 0)].tolist()

    return {
        "true_binding_residues": true_binding_residues,
        "correct_binding": correct_binding,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "total_true_binding": len(true_binding_residues),
        "total_correct": len(correct_binding),
        "total_fp": len(false_positive),
        "total_fn": len(false_negative)
    }


# --------------------------
# 4. 核心函数：找到Top-K蛋白质并保存详细分析结果
# --------------------------
def find_top_predicted_proteins_with_details(model_path, top_k=10, ablation_type=None, threshold=0.5):
    """
    找到测试集中预测效果最好的Top-K蛋白质，并保存残基级详细分析结果
    :param model_path: 模型文件路径
    :param top_k: 要展示/保存的最优蛋白质数量
    :param ablation_type: 消融类型（可选）
    :param threshold: 预测阈值（默认0.5）
    :return: Top-K蛋白质详细信息列表
    """
    print("=" * 80)
    print(f"=== Finding Top {top_k} Predicted Proteins (with residue details) ===")
    print(f"📌 Model path: {model_path}")
    print(f"📌 Prediction threshold: {threshold}")
    print("=" * 80)

    # Step 1: 加载测试集
    print(f"\n📌 Loading test set data...")
    test_loader = get_binding_site_dataloader(split="test")
    total_proteins = len(test_loader.dataset)
    print(f"✅ Test set loaded: {total_proteins} protein sequences")

    # Step 2: 加载模型
    model = load_model(ablation_type=ablation_type, custom_model_path=model_path)
    model.eval()

    # Step 3: 遍历测试集，计算AUC并保存残基详情
    protein_data = []  # 存储所有蛋白质的完整信息
    with torch.no_grad(), tqdm(test_loader, desc="Processing proteins") as pbar:
        for batch in pbar:
            # 模型预测
            loss, pred_list = model(batch)
            batch_labels = batch["binding_site_label"]
            protein_names = batch["pdb_id"]  # 获取真实PDB ID

            # 逐蛋白质处理
            for pred, target, pname in zip(pred_list, batch_labels, protein_names):
                # 数据预处理
                pred_prob = pred.squeeze(-1).cpu().detach().numpy()
                true_label = target.cpu().numpy()
                valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)

                # 计算AUC
                pred_valid = pred_prob[valid_mask]
                target_valid = true_label[valid_mask]
                if len(np.unique(target_valid)) == 2:
                    auc_val = roc_auc_score(target_valid, pred_valid) * 100
                else:
                    auc_val = 50.0

                # 分析残基预测详情
                residue_analysis = analyze_residue_predictions(
                    pred_prob, true_label, valid_mask, threshold
                )

                # 保存该蛋白质的完整信息
                protein_data.append({
                    "pdb_id": pname,
                    "auc": auc_val,
                    "residue_details": residue_analysis
                })

    # Step 4: 按AUC排序，取Top-K
    if not protein_data:
        raise ValueError("❌ No protein data found!")

    # 按AUC降序排序
    protein_data_sorted = sorted(protein_data, key=lambda x: x["auc"], reverse=True)
    top_k_proteins = protein_data_sorted[:top_k]

    # Step 5: 控制台打印Top-K概览
    print("\n" + "=" * 80)
    print(f"🏆 Top {top_k} 预测效果最优的蛋白质（概览）：")
    print("-" * 80)
    print(f"{'排名':<6}{'PDB ID':<15}{'AUC (%)':<10}{'真实结合位点数':<15}")
    print("-" * 80)
    for i, p in enumerate(top_k_proteins, 1):
        print(f"{i:<6}{p['pdb_id']:<15}{p['auc']:.2f}{p['residue_details']['total_true_binding']:<15}")
    print("=" * 80)

    # Step 6: 保存详细结果到文件
    save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "top_proteins_residue_analysis.csv")
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            "排名", "PDB ID", "AUC(%)",
            "真实结合位点序号", "正确预测的结合残基",
            "误判为结合的非结合残基", "漏判的真实结合残基",
            "真实结合位点数", "正确预测数", "误判数", "漏判数"
        ])

        # 写入Top-K蛋白质的详细数据
        for i, p in enumerate(top_k_proteins, 1):
            rd = p["residue_details"]
            # 处理空列表（转为空字符串）
            true_binding = ",".join(map(str, rd["true_binding_residues"])) if rd["true_binding_residues"] else ""
            correct = ",".join(map(str, rd["correct_binding"])) if rd["correct_binding"] else ""
            fp = ",".join(map(str, rd["false_positive"])) if rd["false_positive"] else ""
            fn = ",".join(map(str, rd["false_negative"])) if rd["false_negative"] else ""

            writer.writerow([
                i,
                p["pdb_id"],
                f"{p['auc']:.2f}",
                true_binding,
                correct,
                fp,
                fn,
                rd["total_true_binding"],
                rd["total_correct"],
                rd["total_fp"],
                rd["total_fn"]
            ])

    print(f"\n✅ 详细分析结果已保存到：{save_path}")
    print("=" * 80)

    return top_k_proteins


# --------------------------
# 5. 执行入口
# --------------------------
if __name__ == "__main__":
    try:
        # --------------------------
        # 配置区（可自定义）
        # --------------------------
        TARGET_MODEL_PATH = './bs_checkpoints/best_model3-7.pth'  # 你的模型路径
        TOP_K = 100  # 要分析的最优蛋白质数量
        PREDICTION_THRESHOLD = 0.5  # 预测阈值（可调整，如0.4、0.6）

        # 执行分析
        top_proteins_details = find_top_predicted_proteins_with_details(
            model_path=TARGET_MODEL_PATH,
            top_k=TOP_K,
            ablation_type=None,
            threshold=PREDICTION_THRESHOLD
        )

    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        print(f"\n排障建议:")
        print(f"   1. 检查模型路径 '{TARGET_MODEL_PATH}' 是否存在")
        print(f"   2. 确保dataloader能正确返回 'pdb_id' 字段")
        print(f"   3. Windows用户请将dataloader的num_workers设为0")
        print(f"   4. 验证测试集包含有效的结合位点标签")