import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 导入核心依赖（适配BindingSitePredictor完整模型）
# --------------------------
from model import (
    MODEL_CONFIG,
    BindingSitePredictor,  # 加载完整预测模型
    # 不再直接导入detect_checkpoint_and_load_model，而是重新定义
)
from dataloader import get_binding_site_dataloader
from pretrain import PRETRAIN_CONFIG  # 复用预训练配置

# ===================== t-SNE可视化配置 =====================
TSNE_CONFIG = {
    "target_sample_num": 10000,
    "pca_dim": 50,
    "tsne_perplexity": 50,  # 增大perplexity，适配10000样本的全局分布
    "tsne_n_iter": 5000,  # 增加迭代次数，让分布更稳定
    "random_seed": 42,
    "figure_size": (12, 8),
    "dpi": 300,
    "colors": {"binding": "#ff7f0e", "non_binding": "#1f77b4"},
    "save_results": True,
    # 新增t-SNE优化参数
    "early_exaggeration": 8.0,  # 降低早exaggeration，避免簇过度分离
    "metric": "euclidean",  # 改用欧氏距离，更适合连续特征的距离度量
    "learning_rate": 200  # 固定学习率，提升稳定性
}

# 设置随机种子
np.random.seed(TSNE_CONFIG["random_seed"])
torch.manual_seed(TSNE_CONFIG["random_seed"])


# --------------------------
# 核心修复：重新定义detect_checkpoint_and_load_model函数
# 避开对best_metric的引用，不修改原model.py
# --------------------------
def detect_checkpoint_and_load_model():
    """
    重写版本：移除对best_metric的引用，保留核心功能
    """
    # 1. 检查模型保存目录是否存在
    if not os.path.exists(MODEL_CONFIG["checkpoint_dir"]):
        print(f"⚠️  模型保存目录 {MODEL_CONFIG['checkpoint_dir']} 不存在")
        print("   → 未检测到任何断点或训练模型，将使用初始化模型进行测试")
        return BindingSitePredictor().to(MODEL_CONFIG["device"]), False

    # 2. 定义模型文件路径
    best_model_path = os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["best_model_name"])
    latest_model_path = os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["latest_model_name"])

    # 3. 检测并加载模型
    checkpoint_exists = False
    model = BindingSitePredictor().to(MODEL_CONFIG["device"])

    if os.path.exists(best_model_path):
        # 加载最优模型（移除best_metric相关打印）
        model.load_state_dict(torch.load(best_model_path, map_location=MODEL_CONFIG["device"]))
        checkpoint_exists = True
        print(f"✅ 检测到最优模型文件：{best_model_path}")

    elif os.path.exists(latest_model_path):
        # 加载断点模型
        checkpoint = torch.load(latest_model_path, map_location=MODEL_CONFIG["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_exists = True
        print(f"✅ 检测到训练断点文件：{latest_model_path}")
        # 移除对best_metric的引用，保留核心信息
        if "current_epoch" in checkpoint:
            print(f"   → 断点对应epoch：{checkpoint['current_epoch']}")
        if "last_train_loss" in checkpoint:
            print(f"   → 上一轮训练损失：{checkpoint['last_train_loss']:.4f}")

    else:
        print("⚠️  未检测到训练断点或最优模型")
        print("   → 将使用初始化模型进行测试")

    # 4. 返回模型（测试模式）
    model.eval()
    return model, checkpoint_exists


def load_complete_model():
    """
    加载完整的BindingSitePredictor模型（含预训练权重）
    优先加载最优模型，其次加载断点模型
    """
    model, checkpoint_exists = detect_checkpoint_and_load_model()
    if not checkpoint_exists:
        print("⚠️  警告：未检测到训练模型，将使用随机初始化模型！")
        # 仍继续执行，仅提示警告

    # 设置模型为评估模式
    model.eval()
    # 冻结所有参数（仅特征提取，不训练）
    for param in model.parameters():
        param.requires_grad = False

    return model


def extract_gcn_features():
    """
    提取动态掩码GCN处理后的最终节点特征（替代原aug_emb）
    保留原有的数据加载和标签收集逻辑
    """
    device = MODEL_CONFIG["device"]

    # 加载完整模型（含ProtT5 + ContactGuidedAugment + DynamicMaskGCN）
    print(f"\n📌 加载完整BindingSitePredictor模型...")
    model = load_complete_model()
    print(f"✅ 模型加载完成，设备：{device}")

    # 加载数据集（保持原逻辑）
    print(f"\n📌 加载数据集（get_binding_site_dataloader split='train'）...")
    test_loader = get_binding_site_dataloader(split="train")
    print(f"✅ 数据集加载完成：共{len(test_loader.dataset)}个蛋白质样本")

    # 存储GCN特征和残基标签
    all_gcn_emb = []  # 存储每个残基的GCN后特征
    all_labels = []  # 存储每个残基的标签（1=结合位点，0=非结合位点）
    total_residues = 0
    max_extract = TSNE_CONFIG["target_sample_num"] * 1.2  # 多提取20%用于平衡采样

    with torch.no_grad(), tqdm(test_loader, desc="提取GCN后特征") as pbar:
        for batch in pbar:
            if total_residues >= max_extract:
                break  # 达到提取上限，停止

            # 从batch中获取核心数据
            residue_seqs = batch["residue_sequence"]
            seq_lens = batch["seq_len"].tolist()
            contact_mats = batch["contact_matrix"]
            bs_labels = batch["binding_site_label"]

            # 逐个样本提取GCN特征
            for seq, seq_len, contact_mat, bs_label in zip(
                    residue_seqs, seq_lens, contact_mats, bs_labels
            ):
                # 1. 提取ProtT5嵌入（复用模型内置的ProtT5模块）
                protT5_emb = model.protT5(residue_seqs=[seq], seq_lens=[seq_len])[0].to(device)

                # 2. 嵌入增强生成aug_emb
                contact_mat = contact_mat.to(device)
                aug_emb = model.emb_augment(protT5_emb, contact_mat)

                # 3. 动态掩码GCN处理，获取最终节点特征（核心修改）
                gcn_emb = model.dynamic_gcn.get_final_node_features(aug_emb, contact_mat)

                # 4. 收集残基标签（按实际序列长度截取）
                bs_label = bs_label[:seq_len].to(device)

                # 5. 控制总提取量
                if total_residues + seq_len > max_extract:
                    take_len = int(max_extract - total_residues)
                    gcn_emb = gcn_emb[:take_len]
                    bs_label = bs_label[:take_len]
                    seq_len = take_len

                # 6. 保存特征和标签
                all_gcn_emb.append(gcn_emb.cpu().numpy())
                all_labels.extend(bs_label.cpu().numpy().tolist())
                total_residues += seq_len

                pbar.set_postfix({"已提取残基数": total_residues})

    # 合并特征和标签
    all_gcn_emb = np.concatenate(all_gcn_emb, axis=0)
    all_labels = np.array(all_labels[:len(all_gcn_emb)])

    print(f"\n✅ GCN特征提取完成：")
    print(f"   - 总残基数：{all_gcn_emb.shape[0]}")
    print(f"   - 特征维度：{all_gcn_emb.shape[1]}（对应MODEL_CONFIG['node_dim']={MODEL_CONFIG['node_dim']}）")
    print(f"   - 结合位点残基：{np.sum(all_labels)} ({np.sum(all_labels) / len(all_labels) * 100:.2f}%)")
    print(
        f"   - 非结合位点残基：{len(all_labels) - np.sum(all_labels)} ({(len(all_labels) - np.sum(all_labels)) / len(all_labels) * 100:.2f}%)")

    return all_gcn_emb, all_labels


def balanced_sample(features, labels):
    """
    平衡采样，严格保留10000个残基（解决类别不平衡问题）
    """
    # 分离两类样本
    binding_mask = labels == 1
    non_binding_mask = labels == 0
    binding_feats = features[binding_mask]
    binding_labels = labels[binding_mask]
    non_binding_feats = features[non_binding_mask]
    non_binding_labels = labels[non_binding_mask]

    # 计算样本数量
    n_binding = len(binding_feats)
    n_non_binding = len(non_binding_feats)

    # 平衡策略：两类样本尽可能各占50%，总数量=10000
    target_per_class = TSNE_CONFIG["target_sample_num"] // 2
    binding_sample = min(target_per_class, n_binding)
    non_binding_sample = TSNE_CONFIG["target_sample_num"] - binding_sample

    # 若某类样本不足，用另一类补充
    if non_binding_sample > n_non_binding:
        non_binding_sample = n_non_binding
        binding_sample = TSNE_CONFIG["target_sample_num"] - non_binding_sample

    # 随机采样
    binding_indices = np.random.choice(n_binding, binding_sample, replace=False)
    non_binding_indices = np.random.choice(n_non_binding, non_binding_sample, replace=False)

    # 合并并打乱
    sampled_feats = np.concatenate([
        binding_feats[binding_indices],
        non_binding_feats[non_binding_indices]
    ], axis=0)
    sampled_labels = np.concatenate([
        binding_labels[binding_indices],
        non_binding_labels[non_binding_indices]
    ], axis=0)

    # 打乱顺序
    shuffle_indices = np.random.permutation(len(sampled_feats))
    sampled_feats = sampled_feats[shuffle_indices]
    sampled_labels = sampled_labels[shuffle_indices]

    print(f"\n🔍 平衡采样后（10000个残基）：")
    print(f"   - 结合位点残基：{np.sum(sampled_labels)} ({np.sum(sampled_labels) / len(sampled_labels) * 100:.2f}%)")
    print(
        f"   - 非结合位点残基：{len(sampled_labels) - np.sum(sampled_labels)} ({(len(sampled_labels) - np.sum(sampled_labels)) / len(sampled_labels) * 100:.2f}%)")

    return sampled_feats, sampled_labels


def preprocess_features(features):
    """优化：保留更多PCA方差，确保特征区分性"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_features = features_scaled.shape[1]
    if n_features > TSNE_CONFIG["pca_dim"]:
        # 先计算PCA累计方差，确保至少保留95%的信息
        pca_full = PCA(random_state=TSNE_CONFIG["random_seed"])
        pca_full.fit(features_scaled)
        cum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        # 找到能保留95%方差的最小维度
        min_dim = np.argmax(cum_variance >= 0.95) + 1
        final_pca_dim = max(min_dim, TSNE_CONFIG["pca_dim"])  # 至少50维

        print(f" PCA优化：{n_features}维 → {final_pca_dim}维（保留≥95%方差）")
        pca = PCA(n_components=final_pca_dim, random_state=TSNE_CONFIG["random_seed"])
        features_processed = pca.fit_transform(features_scaled)
        print(f" PCA累计解释方差比例：{np.sum(pca.explained_variance_ratio_):.4f}")
    else:
        features_processed = features_scaled
        print(f"ℹ 特征维度{n_features}≤{TSNE_CONFIG['pca_dim']}，跳过PCA降维")

    return features_processed


def visualize_tsne(features, labels):
    """
    t-SNE可视化GCN后特征（10000个残基）
    """
    # 1. 平衡采样10000个残基
    sampled_feats, sampled_labels = balanced_sample(features, labels)
    assert len(sampled_feats) == TSNE_CONFIG["target_sample_num"], f"采样失败，实际残基数{len(sampled_feats)}≠10000"

    # 2. 特征预处理
    sampled_feats = preprocess_features(sampled_feats)

    # 3. t-SNE降维
    print(f"\n 开始t-SNE降维（10000个残基，perplexity={TSNE_CONFIG['tsne_perplexity']}）")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_CONFIG["tsne_perplexity"],
        n_iter=TSNE_CONFIG["tsne_n_iter"],
        random_state=TSNE_CONFIG["random_seed"],
        verbose=1,
        init='pca',  # 保持PCA初始化，提升稳定性
        learning_rate=TSNE_CONFIG["learning_rate"],
        n_jobs=-1,
        early_exaggeration=TSNE_CONFIG["early_exaggeration"],
        metric=TSNE_CONFIG["metric"],
        angle=0.5
    )
    tsne_results = tsne.fit_transform(sampled_feats)

    # 4. 保存结果（可选）
    if TSNE_CONFIG["save_results"]:
        save_dir = PRETRAIN_CONFIG["checkpoint_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存t-SNE坐标
        tsne_df = pd.DataFrame({
            "tsne_1": tsne_results[:, 0],
            "tsne_2": tsne_results[:, 1],
            "label": ["binding" if l == 1 else "non_binding" for l in sampled_labels],
            "feature_type": "dynamic_mask_gcn_emb"
        })
        tsne_csv_path = os.path.join(save_dir, "tsne_10k_gcn_emb_results.csv")
        tsne_df.to_csv(tsne_csv_path, index=False)
        print(f"✅ t-SNE坐标已保存至：{tsne_csv_path}")

    # 5. 可视化绘图
    plt.figure(figsize=TSNE_CONFIG["figure_size"], dpi=TSNE_CONFIG["dpi"])

    # 分离两类残基
    binding_mask = sampled_labels == 1
    tsne_binding = tsne_results[binding_mask]
    tsne_non_binding = tsne_results[~binding_mask]

    # 绘制散点图
    plt.scatter(
        tsne_non_binding[:, 0],
        tsne_non_binding[:, 1],
        c=TSNE_CONFIG["colors"]["non_binding"],
        label="Non-binding Residues",
        alpha=0.7,
        s=12,
        edgecolors='none',
        rasterized=True
    )
    plt.scatter(
        tsne_binding[:, 0],
        tsne_binding[:, 1],
        c=TSNE_CONFIG["colors"]["binding"],
        label="Binding Residues",
        alpha=0.7,
        s=12,
        edgecolors='none',
        rasterized=True
    )

    # 图表美化
    plt.legend(loc='best', frameon=True, framealpha=0.9, fontsize=12)
    plt.title(f't-SNE Visualization (Dynamic Mask GCN Features)',
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # 添加统计信息文本框
    stats_text = (
        f'Binding Residues: {len(tsne_binding)} ({len(tsne_binding) / len(sampled_labels) * 100:.1f}%)\n'
        f'Non-binding Residues: {len(tsne_non_binding)} ({len(tsne_non_binding) / len(sampled_labels) * 100:.1f}%)\n'
        f'Feature: Dynamic Mask GCN Output (node_dim={MODEL_CONFIG["node_dim"]})'
    )
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # 保存图片
    save_path = os.path.join(PRETRAIN_CONFIG["checkpoint_dir"], "tsne_10k_gcn_emb.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=TSNE_CONFIG["dpi"], bbox_inches='tight', facecolor='white')
    print(f"✅ 可视化图片已保存至：{save_path}")

    plt.show()
    plt.close()


def filter_invalid_features(features, labels):
    """
    过滤无效特征（全零向量/NaN/Inf）
    """
    # 过滤全零特征
    valid_mask = ~(np.all(features == 0, axis=1))
    # 过滤NaN/Inf
    valid_mask = valid_mask & np.isfinite(features).all(axis=1)

    features_valid = features[valid_mask]
    labels_valid = labels[valid_mask]

    n_filtered = len(features) - len(features_valid)
    print(f"\n🔍 特征过滤结果：")
    print(f"   - 原始残基数：{len(features)}")
    print(f"   - 过滤后残基数：{len(features_valid)}")
    print(f"   - 过滤无效残基数：{n_filtered}")

    return features_valid, labels_valid


def main():
    try:
        print("=" * 80)
        print("=== t-SNE Visualization: Dynamic Mask GCN Features ===")
        print("=== 使用get_binding_site_dataloader(split='train')加载训练集 ===")
        print("=" * 80)

        # 1. 提取GCN后特征和残基标签
        gcn_features, residue_labels = extract_gcn_features()

        # 2. 过滤无效特征
        gcn_features, residue_labels = filter_invalid_features(gcn_features, residue_labels)

        # 3. 检查有效样本数量
        if len(gcn_features) < TSNE_CONFIG["target_sample_num"]:
            raise ValueError(f"❌ 有效残基数{len(gcn_features)}<10000，无法完成映射！")

        # 4. t-SNE可视化
        visualize_tsne(gcn_features, residue_labels)

        print("\n🎉 所有流程执行完成！")
        print(f"📌 核心信息：")
        print(f"   - 映射残基数：10000个")
        print(f"   - 特征类型：动态掩码GCN处理后的最终节点特征（node_dim={MODEL_CONFIG['node_dim']}）")
        print(f"   - 数据集加载：get_binding_site_dataloader(split='train')")
        print(f"   - 结果文件：t-SNE图 + 坐标CSV（保存于{PRETRAIN_CONFIG['checkpoint_dir']}）")

    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()