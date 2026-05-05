import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# 从eva.py导入模型加载函数（避免使用有问题的detect_checkpoint_and_load_model）
from eva import load_model
from dataloader import get_binding_site_dataloader



def predict_top10_samples():
    """在测试集上前10个样本上进行预测，输出真实标签与预测概率"""
    print("=== 测试集前10样本预测结果展示 ===")
    print(f"📌 配置信息：")
    print(f"   - 预测结果包含：样本ID、残基位置、真实标签(0/1)、预测概率")

    # 1. 加载模型（使用eva.py中的load_model函数，避免KeyError）
    model, model_path, model_type = load_model()
    if model_type == "random":
        print("⚠️  警告：使用随机初始化模型，预测结果无意义")
    else:
        print(f"✅ 模型加载完成：{model_path}")
    model.eval()

    # 2. 加载测试集数据加载器
    test_loader = get_binding_site_dataloader(split="test")
    print(f"✅ 测试集加载完成：共{len(test_loader.dataset)}个样本")

    # 3. 存储前10个样本的预测结果
    results = []
    sample_count = 0
    max_samples = 10

    with torch.no_grad(), tqdm(
            test_loader,
            total=len(test_loader),
            desc="获取预测结果"
    ) as pbar:
        for batch in pbar:
            if sample_count >= max_samples:
                break  # 只处理前10个样本

            # 模型预测（适配BindingSitePredictor的forward输出）
            _, pred_list = model(batch)
            pdb_ids = batch["pdb_id"]
            labels = batch["binding_site_label"]
            seq_lens = batch["seq_len"].tolist()

            # 处理批次中的每个样本
            for i in range(len(pdb_ids)):
                if sample_count >= max_samples:
                    break

                pdb_id = pdb_ids[i]
                seq_len = seq_lens[i]
                true_label = labels[i].cpu().numpy()  # 真实标签(0/1)
                pred_prob = pred_list[i].squeeze(-1).cpu().numpy()  # 预测概率

                # 记录每个残基的结果
                for residue_pos in range(seq_len):
                    results.append({
                        "样本ID": pdb_id,
                        "残基位置(0-based)": residue_pos,
                        "真实标签": int(true_label[residue_pos]),
                        "预测概率": round(float(pred_prob[residue_pos]), 4)
                    })

                sample_count += 1
                print(f"✅ 已处理样本 {sample_count}/{max_samples}：{pdb_id}（残基数：{seq_len}）")

    # 4. 保存结果到CSV
    if results:
        result_df = pd.DataFrame(results)
        output_path = "top10_samples_predictions.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\n=== 预测结果汇总 ===")
        print(f"📊 前10个样本的预测结果已保存至：{output_path}")

        # 打印前5行示例
        print("\n📌 结果示例（前5行）：")
        print(result_df.head().to_string(index=False))
    else:
        print("❌ 未获取到任何预测结果")


if __name__ == "__main__":
    predict_top10_samples()