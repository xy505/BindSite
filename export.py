import os
import numpy as np
import torch
from tqdm import tqdm
import csv

import warnings
warnings.filterwarnings("ignore")

# --------------------------
# 1. 导入核心依赖 + 强制修改测试集路径为 tmp.csv
# --------------------------
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
from train import (
    TRAIN_CONFIG,
    set_ablation_mode
)

# ✅ 核心修改：强制把测试集CSV改为 ./data/tmp.csv
DATALOADER_CONFIG["test_csv_path"] = "./data/tmp.csv"

# --------------------------
# 2. 模型加载函数（完全保留你的原版代码）
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
# 3. 新增：生成预测01序列函数（匹配你的数据格式）
# --------------------------
def generate_01_sequence(pred_prob, valid_mask, threshold=0.5):
    """生成与蛋白质长度一致的01预测序列"""
    pred_binary = []
    for prob, mask in zip(pred_prob, valid_mask):
        if not mask:
            pred_binary.append('*')  # 无效填充位
        else:
            pred_binary.append('1' if prob >= threshold else '0')
    return ''.join(pred_binary)

# --------------------------
# 4. 核心函数：读取tmp.csv 输出预测01序列CSV
# --------------------------
def predict_tmp_csv_and_save_01(model_path, threshold=0.5, ablation_type=None):
    print("=" * 80)
    print(f"=== 预测 tmp.csv 数据集，输出01序列 ===")
    print(f"📌 测试集文件: {DATALOADER_CONFIG['test_csv_path']}")
    print(f"📌 模型路径: {model_path}")
    print(f"📌 预测阈值: {threshold}")
    print("=" * 80)

    # 加载【修改后的tmp.csv测试集】
    print(f"\n📌 加载 tmp.csv 数据...")
    test_loader = get_binding_site_dataloader(split="test")
    total_proteins = len(test_loader.dataset)
    print(f"✅ 加载完成：共 {total_proteins} 个蛋白质")

    # 加载模型
    model = load_model(ablation_type=ablation_type, custom_model_path=model_path)
    model.eval()

    # 存储预测结果
    results = []

    # 开始预测
    with torch.no_grad(), tqdm(test_loader, desc="预测中") as pbar:
        for batch in pbar:
            loss, pred_list = model(batch)
            protein_names = batch["pdb_id"]
            batch_labels = batch["binding_site_label"]

            # 逐蛋白生成01序列
            for pred, target, pname in zip(pred_list, batch_labels, protein_names):
                pred_prob = pred.squeeze(-1).cpu().detach().numpy()
                true_label = target.cpu().numpy()
                valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)

                # 生成01预测序列
                pred_01 = generate_01_sequence(pred_prob, valid_mask, threshold)
                results.append([pname, pred_01])

    # 保存为CSV文件
    save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "tmp_csv_prediction_01.csv")
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Protein ID", "Predicted 01 Sequence"])
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"✅ 预测完成！")
    print(f"📊 总蛋白质数量：{len(results)}")
    print(f"📁 01序列文件已保存到：{save_path}")
    print("=" * 80)
    return results

# --------------------------
# 5. 执行入口（无需修改其他代码）
# --------------------------
if __name__ == "__main__":
    try:
        # 配置（和你原版一致）
        TARGET_MODEL_PATH = './bs_checkpoints/best_model3-7.pth'
        PREDICTION_THRESHOLD = 0.5

        # 执行预测 tmp.csv
        predict_tmp_csv_and_save_01(
            model_path=TARGET_MODEL_PATH,
            threshold=PREDICTION_THRESHOLD,
            ablation_type=None
        )

    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        print(f"\n排障建议:")
        print(f"   1. 检查 ./data/tmp.csv 文件是否存在")
        print(f"   2. 检查模型路径是否正确")
        print(f"   3. Windows用户请将num_workers设为0")