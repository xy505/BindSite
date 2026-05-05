import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils import compute_residue_contact_matrix  # 复用《方案》接触矩阵函数

# 全局配置（新增训练集和测试集CSV路径配置）
CONFIG = {
    "pdb_dir": "./data/pdb",  # 《方案》中PDBbind数据集根目录
    "train_csv_path": "./data/train.csv",  # 训练集CSV路径
    "test_csv_path": "./data/test.csv",  # 测试集CSV路径
    "contact_threshold": 8.0,  # 《方案》中残基接触判定阈值（8Å，0.1节）
    "batch_size": 16,  # 《方案》中模型训练批次大小（6节）
    "num_workers": 2,  # 《方案》中数据加载线程数（匹配CPU核心）
    "train_ratio": 0.75,  # 仅用于训练集内部划分验证集
    "val_ratio": 0.125  # 仅用于训练集内部划分验证集
}


# 自定义collate_fn：按实际残基长度组织batch（不统一维度，仅堆叠元数据）
def custom_collate_fn(batch):
    """
    不做任何Padding，仅收集batch内各样本的实际数据（维度与残基个数一致）
    返回列表形式的batch，适配后续模型逐样本处理逻辑
    """
    return {
        "pdb_id": [sample["pdb_id"] for sample in batch],  # 样本标识列表
        "residue_sequence": [sample["residue_sequence"] for sample in batch],  # 实际长度序列列表
        "contact_matrix": [sample["contact_matrix"] for sample in batch],  # 实际维度接触矩阵列表（[seq_len, seq_len]）
        "binding_site_label": [sample["binding_site_label"] for sample in batch],  # 实际长度标签列表（[seq_len]）
        "seq_len": torch.tensor([sample["seq_len"] for sample in batch], dtype=torch.int32)  # 各样本实际残基个数
    }


# Dataset类：按实际残基长度输出样本（无Padding，无截断）
class BindingSiteDataset(Dataset):
    def __init__(self, split="train"):
        # 根据split选择不同的CSV路径
        if split in ["train", "val"]:
            csv_path = CONFIG["train_csv_path"]  # 训练集和验证集共用训练CSV
        elif split == "test":
            csv_path = CONFIG["test_csv_path"]  # 测试集使用测试CSV
        else:
            raise ValueError("split必须为'train'/'val'/'test'（《方案》2节实验设计）")

        # 加载对应CSV数据（强制字符串类型避免空值）
        self.csv_data = pd.read_csv(
            csv_path,
            dtype={"PDB": str, "Sequence": str, "BS": str}
        )

        # 数据集划分（仅对训练集进行内部划分，测试集不划分）
        np.random.seed(42)
        total_samples = len(self.csv_data)
        shuffled_indices = np.random.permutation(total_samples)
        train_size = int(total_samples * CONFIG["train_ratio"])
        val_size = int(total_samples * CONFIG["val_ratio"])

        # 按split分配样本索引
        if split == "train":
            self.selected_indices = shuffled_indices[:train_size]
        elif split == "val":
            self.selected_indices = shuffled_indices[train_size:train_size + val_size]
        elif split == "test":
            self.selected_indices = shuffled_indices  # 测试集使用全部样本
        else:
            raise ValueError("split必须为'train'/'val'/'test'（《方案》2节实验设计）")

    def _parse_bs_label(self, bs_str, seq_len):
        """《方案》2节标签解析：生成与实际残基长度一致的结合位点标签"""
        label = np.zeros(seq_len, dtype=np.float32)
        bs_segments = [seg.strip() for seg in bs_str.split(",") if seg.strip()]
        for seg in bs_segments:
            if seg.isdigit():
                pos = int(seg)   # 转为0-based索引（匹配模型输入逻辑）
                if 0 <= pos < seq_len:
                    label[pos] = 1.0  # 标记结合位点残基
        return label

    def __len__(self):
        return len(self.selected_indices)  # 样本总数（《方案》2节数据规模统计）

    def __getitem__(self, idx):
        """按实际残基长度返回单样本，无Padding、无截断（完全匹配蛋白质真实结构）"""
        sample_idx = self.selected_indices[idx]
        sample = self.csv_data.iloc[sample_idx]

        # 提取核心数据（《方案》2节实验数据字段）
        pdb_id = sample["PDB"]
        raw_residue_seq = sample["Sequence"]  # 原始残基序列（无截断）
        bs_str = sample["BS"]  # 结合位点标签字符串

        # 实际残基个数（与蛋白质真实长度一致）
        actual_seq_len = len(raw_residue_seq)

        # 计算残基接触矩阵（《方案》0.1节核心数据，维度与残基个数一致）
        pdb_folder = os.path.join(CONFIG["pdb_dir"], pdb_id)
        pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith("_protein.pdb")]
        if not pdb_files:
            raise FileNotFoundError(f"{pdb_folder}中未找到 *_protein.pdb文件（《方案》数据要求）")
        contact_matrix, _ = compute_residue_contact_matrix(
            os.path.join(pdb_folder, pdb_files[0])
        )

        # 接触矩阵维度对齐（确保与实际残基个数一致，《方案》0.1节空间约束要求）
        if len(contact_matrix) != actual_seq_len:
            min_dim = min(len(contact_matrix), actual_seq_len)
            contact_matrix = contact_matrix[:min_dim, :min_dim]
            raw_residue_seq = raw_residue_seq[:min_dim]
            actual_seq_len = min_dim  # 更新实际长度（确保数据一致性）

        # 生成结合位点标签（与实际残基长度一致，《方案》2节标签格式）
        label = self._parse_bs_label(bs_str, actual_seq_len)

        # 返回单样本（所有字段维度与实际残基个数一致）
        return {
            "pdb_id": pdb_id,
            "residue_sequence": raw_residue_seq,  # 实际长度：[actual_seq_len]
            "contact_matrix": torch.tensor(contact_matrix, dtype=torch.float32),
            # 实际维度：[actual_seq_len, actual_seq_len]
            "binding_site_label": torch.tensor(label, dtype=torch.float32),  # 实际长度：[actual_seq_len]
            "seq_len": actual_seq_len  # 实际残基个数（用于模型定位有效数据）
        }


# DataLoader生成函数（《方案》6节实验流程：关联无Padding的collate_fn）
def get_binding_site_dataloader(split="train"):
    dataset = BindingSiteDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],  # 《方案》6节批次大小
        shuffle=(split == "train"),  # 训练集打乱（《方案》6节数据增强）
        num_workers=CONFIG["num_workers"],  # 多线程加载（《方案》6节效率优化）
        pin_memory=True,  # 内存锁定（加速GPU数据传输）
        collate_fn=custom_collate_fn  # 关键：按实际长度组织batch
    )


# 测试函数：新增打印首个样本ID的逻辑
if __name__ == "__main__":
    try:
        train_loader = get_binding_site_dataloader(split="train")
        print(f"=== 《方案》Dataloader测试（无统一维度） ===")
        print(f"总批次数量：{len(train_loader)}（批次大小：{CONFIG['batch_size']}）")
        print(f"训练集总样本数：{len(train_loader.dataset)}")

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"\n【第 {batch_idx + 1} 个batch详细信息】")
                # 新增：打印首个样本的PDB ID
                print(f"1. 首个样本PDB ID：{batch['pdb_id'][0]}")  # 关键新增行
                print(f"2. 各样本实际残基个数：{batch['seq_len'].tolist()}")

                for i in range(CONFIG["batch_size"]):
                    seq_len = batch["seq_len"][i].item()
                    print(f"   - 样本{i + 1}（PDB: {batch['pdb_id'][i]}）：")  # 也可在循环中打印每个样本ID
                    print(f"     残基序列长度：{len(batch['residue_sequence'][i])} → 预期={seq_len}")
                    print(f"     接触矩阵维度：{batch['contact_matrix'][i].shape} → 预期=({seq_len},{seq_len})")
                    print(f"     标签长度：{batch['binding_site_label'][i].shape[0]} → 预期={seq_len}")

                first_sample_label = batch["binding_site_label"][0]
                first_seq_len = batch["seq_len"][0].item()
                binding_site_pos = torch.where(first_sample_label == 1.0)[0]  # 无需加1，直接展示0-based位置
                print(f"\n3. 首个样本结合位点（0-based）：{binding_site_pos.tolist()}")

                break

        test_loader = get_binding_site_dataloader(split="test")
        print(f"\n测试集总样本数：{len(test_loader.dataset)}")

        print("\n✅ Dataloader测试完成")
    except Exception as e:
        print(f"❌ Dataloader错误：{str(e)}")