import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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
# 2. 核心函数：单独运行单个模型并保存序列级AUC结果
# --------------------------
def load_model(ablation_type=None, custom_model_path=None):
    """
    Enhanced model loading: support manual model path specification
    """
    model = BindingSitePredictor().to(MODEL_CONFIG["device"])
    model = set_ablation_mode(model, ablation_type)

    # Prioritize manually specified model path
    if custom_model_path is not None and os.path.exists(custom_model_path):
        model.load_state_dict(torch.load(custom_model_path, map_location=MODEL_CONFIG["device"]))
        model.eval()
        print(f"✅ Loaded manually specified model: {custom_model_path}")
        return model, custom_model_path, "custom_specified"

    # Load with original logic if no custom path
    ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
    best_model_filename = f"{MODEL_CONFIG['best_model_name'].rsplit('.', 1)[0]}{ablation_suffix}.pth"
    best_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], best_model_filename)
    latest_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], TRAIN_CONFIG["resume_checkpoint"])

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=MODEL_CONFIG["device"]))
        model.eval()
        print(f"✅ Loaded best trained model: {best_model_path}")
        return model, best_model_path, "trained_best"
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location=MODEL_CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Loaded resume checkpoint model: {latest_model_path}")
        print(f"   - Resume epoch: {checkpoint['current_epoch']}, Last train loss: {checkpoint['last_train_loss']:.4f}")
        return model, latest_model_path, "trained_latest"
    else:
        model.train(False)
        print(f"⚠️ No trained checkpoint detected")
        print(f"   - Using randomly initialized model for testing")
        return model, "random_initialized", "random"


def run_single_model_and_save(model_path, save_filename, ablation_type=None):
    """
    运行单个模型，计算序列级AUC并保存结果
    :param model_path: 模型路径
    :param save_filename: 保存文件名（如"current_model_auc.csv"）
    :param ablation_type: 消融类型（可选）
    """
    print("=" * 100)
    print(f"=== Running Single Model: {os.path.basename(model_path)} ===")
    print(f"📌 Save result to: {save_filename}")
    print("=" * 100)

    # Step 1: Load test set
    print(f"\n📌 Loading test set data...")
    test_loader = get_binding_site_dataloader(split="test")
    total_proteins = len(test_loader.dataset)
    print(f"✅ Test set loaded successfully: total {total_proteins} protein sequences")

    # Step 2: Load model
    model, _, _ = load_model(ablation_type=ablation_type, custom_model_path=model_path)
    model.eval()

    # Step 3: Calculate sequence-level AUC
    sequence_auc_data = []
    with torch.no_grad(), tqdm(test_loader, desc="Calculating sequence-level AUC") as pbar:
        for batch_idx, batch in enumerate(pbar):
            loss, pred_list = model(batch)
            batch_labels = batch["binding_site_label"]
            # 核心修改：直接从batch中获取真实的pdb_id，不再生成占位符
            protein_names = batch["pdb_id"]  # 从dataloader获取真实PDB ID

            # 逐蛋白质计算AUC
            for pred, target, pname in zip(pred_list, batch_labels, protein_names):
                pred_prob = pred.squeeze(-1).cpu().detach().numpy()
                true_label = target.cpu().numpy()
                valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)
                pred_valid = pred_prob[valid_mask]
                target_valid = true_label[valid_mask]

                if len(np.unique(target_valid)) == 2:
                    auc_val = roc_auc_score(target_valid, pred_valid) * 100
                else:
                    auc_val = 50.0

                sequence_auc_data.append({
                    "Protein_Name": pname,  # 保存真实的PDB ID
                    "AUC(%)": round(auc_val, 2)
                })

    # Step 4: Save results to CSV
    save_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], save_filename)
    df = pd.DataFrame(sequence_auc_data)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\n✅ Result saved to: {save_path}")
    print(f"📊 Summary: Total {len(df)} proteins processed")
    print(f"   - Mean AUC: {df['AUC(%)'].mean():.2f}%")
    print(f"   - Median AUC: {df['AUC(%)'].median():.2f}%")
    print("=" * 100)


# --------------------------
# 5. 执行入口（分两次运行，手动切换模型路径）
# --------------------------
if __name__ == "__main__":
    try:
        # --------------------------
        # 第一次运行：当前模型（修改为你的主模型路径）
        # --------------------------
        CURRENT_MODEL_PATH = './bs_checkpoints/best_model3-5.pth'  # 你的主模型路径
        # CURRENT_SAVE_FILENAME = "current_model_sequence_auc.csv"
        CURRENT_SAVE_FILENAME = "reference_model_sequence_auc.csv"
        run_single_model_and_save(CURRENT_MODEL_PATH, CURRENT_SAVE_FILENAME)

        # --------------------------
        # 第二次运行：参考模型（修改为你的参考模型路径，运行完第一次后再运行第二次）
        # --------------------------
        # REFERENCE_MODEL_PATH = './bs_checkpoints/best_model2.pth'  # 你的参考模型路径
        # REFERENCE_SAVE_FILENAME = "reference_model_sequence_auc.csv"
        # run_single_model_and_save(REFERENCE_MODEL_PATH, REFERENCE_SAVE_FILENAME)

    except Exception as e:
        print(f"\n❌ Execution failed: {str(e)}")
        print(f"   Troubleshooting Steps:")
        print(f"   1. Check if model path is correct")
        print(f"   2. Ensure dataloader returns 'pdb_id' correctly")
        print(f"   3. For Windows, set DataLoader's num_workers to 0")