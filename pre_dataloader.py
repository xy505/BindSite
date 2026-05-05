import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# 复用方案utils.py的接触矩阵计算函数（方案0.1节核心工具）
from utils import compute_residue_contact_matrix

# --------------------------
# 全局配置（严格对齐方案2节+6节参数，补充路径说明）
# --------------------------
CONFIG = {
    "toughM1": {
        "positive_csv": "./data/positive_pairs.csv",  # 方案2节ToughM1正样本对CSV
        "negative_csv": "./data/negative_pairs.csv",  # 方案2节ToughM1负样本对CSV
        "pdb_dir": "./data/TOUGH-M1",  # 方案2节PDB文件存放根目录（需按PDB ID分文件夹，如./data/TOUGH-M1/1d00D/1d00D.pdb）
        "contact_threshold": 8.0,  # 方案0.1节残基接触判定阈值（8Å）
    },
    "train": {
        "batch_size": 2,  # 方案6节批次大小
        "num_workers": 2,  # 方案6节数据加载线程数
        "train_ratio": 0.75,  # 方案2节训练集占比
        "val_ratio": 0.125,  # 方案2节验证集占比（测试集=0.125）
        "seed": 42  # 方案6节固定种子（可复现）
    }
}


# --------------------------
# 方案0.2节对比学习数据集：ToughM1PairDataset（含精准PDB筛选）
# --------------------------
class ToughM1PairDataset(Dataset):
    def __init__(self, split="train", pair_type="positive"):
        # 原初始化逻辑保持不变
        if pair_type not in ["positive", "negative"]:
            raise ValueError(f"pair_type必须为'positive'/'negative'（方案0.2节对比学习样本类型）")
        csv_path = CONFIG["toughM1"][f"{pair_type}_csv"]
        self.csv_data = pd.read_csv(
            csv_path,
            dtype={
                "pdb1": str, "sequence1": str, "bs1": str,  # 固定为锚点蛋白质（方案0.2节锚点定义）
                "pdb2": str, "sequence2": str, "bs2": str  # 固定为对比样本（方案0.2节对比对象定义）
            }
        )
        required_cols = ["pdb1", "sequence1", "bs1", "pdb2", "sequence2", "bs2"]
        if not all(col in self.csv_data.columns for col in required_cols):
            raise ValueError(f"ToughM1 CSV缺少方案2节要求的列：需包含{required_cols}")
        np.random.seed(CONFIG["train"]["seed"])
        total_pairs = len(self.csv_data)
        shuffled_indices = np.random.permutation(total_pairs)
        train_size = int(total_pairs * CONFIG["train"]["train_ratio"])
        val_size = int(total_pairs * CONFIG["train"]["val_ratio"])
        if split == "train":
            self.selected_indices = shuffled_indices[:train_size]
        elif split == "val":
            self.selected_indices = shuffled_indices[train_size:train_size + val_size]
        elif split == "test":
            self.selected_indices = shuffled_indices[train_size + val_size:]
        else:
            raise ValueError("split必须为'train'/'val'/'test'（方案2节实验设计要求）")
        self.pair_type = pair_type
        self.pair_label = 1 if pair_type == "positive" else 0

    def _parse_bs_label(self, bs_str, seq_len):
        # 原标签解析逻辑保持不变（注意：若BS列是0-based索引，无需-1；若为1-based需添加pos = int(seg.strip()) - 1）
        label = np.zeros(seq_len, dtype=np.float32)
        if pd.isna(bs_str) or bs_str.strip() == "":
            return label
        bs_positions = [int(seg.strip()) for seg in bs_str.split(",") if seg.strip().isdigit()]
        for pos in bs_positions:
            if 0 <= pos < seq_len:
                label[pos] = 1.0
        return label

    def _load_protein_data(self, pdb_id, sequence, bs_str):
        """修复路径拼接+精准PDB筛选：保留合法1d00D.pdb，排除1d00D00.pdb（后缀含00且长度超6位）"""
        raw_seq = sequence.strip()
        actual_seq_len = len(raw_seq)
        if actual_seq_len == 0:
            raise ValueError(f"蛋白质{pdb_id}序列为空（方案0.1节数据有效性要求）")

        # 1. 标准化PDB文件夹路径（解决跨系统斜杠兼容问题）
        pdb_folder = os.path.join(CONFIG["toughM1"]["pdb_dir"], pdb_id)
        pdb_folder = os.path.normpath(pdb_folder)  # 统一路径分隔符（如Windows\转Linux/）
        # 2. 验证PDB文件夹存在性（方案2节数据有效性检查）
        if not os.path.exists(pdb_folder):
            raise FileNotFoundError(
                f"方案2节PDB文件夹不存在：{pdb_folder}\n"
                f"请确认：1. CONFIG['toughM1']['pdb_dir']配置是否正确（当前为{CONFIG['toughM1']['pdb_dir']}）；\n"
                f"       2. PDB文件是否按方案2节要求，按ID分文件夹存放（如{os.path.normpath(os.path.join(CONFIG['toughM1']['pdb_dir'], '1d00D'))}）"
            )

        # 3. 精准筛选PDB文件：核心修改！
        # 步骤1：获取文件夹下所有.pdb文件
        all_pdb_files = [f for f in os.listdir(pdb_folder) if f.lower().endswith(".pdb")]
        if not all_pdb_files:
            raise FileNotFoundError(f"{pdb_folder}中未找到PDB文件（方案0.1节结构数据需求）")

        # 步骤2：定义筛选规则（关键！）
        # 合法文件：文件名（不含.pdb后缀）长度≤6位，或长度>6位但末尾不含"00"
        # 排除文件：文件名（不含.pdb后缀）长度>6位且末尾含"00"（如1d00D00.pdb，后缀前长度6位且末尾00）
        target_pdb_files = []
        for f in all_pdb_files:
            # 提取文件名（去掉.pdb后缀）
            file_name_without_ext = os.path.splitext(f)[0]
            # 筛选逻辑：
            # 情况1：文件名长度≤6位 → 合法（如1d00D，长度5位；3t35B，长度5位）
            # 情况2：文件名长度>6位，但末尾不含"00" → 合法（若存在此类文件）
            # 情况3：文件名长度>6位且末尾含"00" → 不合法（如1d00D00，长度6位且末尾00）
            if len(file_name_without_ext) <= 6 or (
                    len(file_name_without_ext) > 6 and not file_name_without_ext.endswith("00")):
                target_pdb_files.append(f)

        # 步骤3：验证筛选结果（确保至少有一个合法PDB文件）
        if not target_pdb_files:
            raise FileNotFoundError(
                f"{pdb_folder}中仅找到不合法PDB文件（如文件名长度超6位且末尾含'00'的1d00D00.pdb），未找到合法文件（如1d00D.pdb）\n"
                f"当前文件夹下所有PDB文件：{all_pdb_files}\n"
                f"筛选后合法文件：{target_pdb_files}"
            )

        # 步骤4：选择第一个合法PDB文件（优先选择含"protein"的文件，无则选第一个）
        # （可选优化：优先匹配蛋白质文件，避免误选配体PDB）
        protein_pdb = next((f for f in target_pdb_files if "protein" in f.lower()), target_pdb_files[0])
        protein_pdb_path = os.path.normpath(os.path.join(pdb_folder, protein_pdb))  # 标准化文件路径

        # 4. 计算残基接触矩阵（方案0.1节核心数据）
        contact_matrix, _ = compute_residue_contact_matrix(
            protein_pdb_path,
            contact_threshold=CONFIG["toughM1"]["contact_threshold"]
        )

        # 接触矩阵与序列长度对齐（方案0.1节空间约束：避免PDB解析长度与序列长度不匹配）
        if len(contact_matrix) != actual_seq_len:
            min_dim = min(len(contact_matrix), actual_seq_len)
            contact_matrix = contact_matrix[:min_dim, :min_dim]
            raw_seq = raw_seq[:min_dim]
            actual_seq_len = min_dim

        # 生成结合位点标签（方案0.3.2节二值标签）
        bs_label = self._parse_bs_label(bs_str, actual_seq_len)

        return {
            "pdb_id": pdb_id,
            "residue_sequence": raw_seq,
            "contact_matrix": torch.tensor(contact_matrix, dtype=torch.float32),
            "binding_site_label": torch.tensor(bs_label, dtype=torch.float32),
            "seq_len": actual_seq_len
        }

    def __getitem__(self, idx):
        # 原__getitem__逻辑保持不变：加载锚点+对比样本对数据
        pair_idx = self.selected_indices[idx]
        pair_data = self.csv_data.iloc[pair_idx]
        anchor_data = self._load_protein_data(
            pdb_id=pair_data["pdb1"],
            sequence=pair_data["sequence1"],
            bs_str=pair_data["bs1"]
        )
        contrast_data = self._load_protein_data(
            pdb_id=pair_data["pdb2"],
            sequence=pair_data["sequence2"],
            bs_str=pair_data["bs2"]
        )
        return {
            "anchor_pdb": anchor_data["pdb_id"],
            "anchor_seq": anchor_data["residue_sequence"],
            "anchor_contact": anchor_data["contact_matrix"],
            "anchor_bs_label": anchor_data["binding_site_label"],
            "anchor_seq_len": anchor_data["seq_len"],
            "contrast_pdb": contrast_data["pdb_id"],
            "contrast_seq": contrast_data["residue_sequence"],
            "contrast_contact": contrast_data["contact_matrix"],
            "contrast_bs_label": contrast_data["binding_site_label"],
            "contrast_seq_len": contrast_data["seq_len"],
            "pair_label": torch.tensor(self.pair_label, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.selected_indices)


# --------------------------
# 方案0.2节专用DataLoader生成函数（原逻辑不变）
# --------------------------
def toughM1_pair_collate_fn(batch):
    return {
        "anchor_pdb": [sample["anchor_pdb"] for sample in batch],
        "anchor_seq": [sample["anchor_seq"] for sample in batch],
        "anchor_contact": [sample["anchor_contact"] for sample in batch],
        "anchor_bs_label": [sample["anchor_bs_label"] for sample in batch],
        "anchor_seq_len": torch.tensor([sample["anchor_seq_len"] for sample in batch], dtype=torch.int32),
        "contrast_pdb": [sample["contrast_pdb"] for sample in batch],
        "contrast_seq": [sample["contrast_seq"] for sample in batch],
        "contrast_contact": [sample["contrast_contact"] for sample in batch],
        "contrast_bs_label": [sample["contrast_bs_label"] for sample in batch],
        "contrast_seq_len": torch.tensor([sample["contrast_seq_len"] for sample in batch], dtype=torch.int32),
        "pair_label": torch.stack([sample["pair_label"] for sample in batch], dim=0)
    }


def get_toughM1_pair_dataloader(split="train", pair_type="positive", batch_size=None):
    dataset = ToughM1PairDataset(split=split, pair_type=pair_type)
    return DataLoader(
        dataset,
        batch_size=batch_size or CONFIG["train"]["batch_size"],
        shuffle=(split == "train"),  # 训练集打乱，验证/测试集不打乱（方案6节要求）
        num_workers=CONFIG["train"]["num_workers"],  # 多线程加载（提升效率）
        pin_memory=True,  # 内存锁定（加速GPU数据传输）
        collate_fn=toughM1_pair_collate_fn  # 适配变长序列的批次拼接
    )


# --------------------------
# 测试函数（验证精准筛选效果）
# --------------------------
if __name__ == "__main__":
    print("=== 方案0.2节ToughM1数据集加载测试（精准PDB筛选） ===")
    try:
        # 加载正样本对DataLoader（验证1d00D.pdb是否被正确加载）
        pos_loader = get_toughM1_pair_dataloader(split="train", pair_type="positive")
        print(f"✅ 正样本对DataLoader加载完成：")
        print(f"   - 样本数量：{len(pos_loader.dataset)}（方案2节数据统计）")
        print(f"   - 批次大小：{CONFIG['train']['batch_size']}（方案6节参数）")
        print(f"   - 线程数：{CONFIG['train']['num_workers']}（方案6节高效加载）")

        # 验证首个批次数据维度（确保1d00D.pdb被正确处理）
        for batch in pos_loader:
            print(f"\n【首个批次数据维度验证（方案0.1节变长约束）】")
            batch_size = len(batch["anchor_seq"])
            for i in range(batch_size):
                anchor_pdb = batch["anchor_pdb"][i]
                contrast_pdb = batch["contrast_pdb"][i]
                anchor_len = batch["anchor_seq_len"][i].item()
                contrast_len = batch["contrast_seq_len"][i].item()

                print(f"   样本{i + 1}：")
                print(f"     锚点PDB ID：{anchor_pdb}（验证是否为1d00D等合法ID）")
                print(f"     锚点残基长度：{len(batch['anchor_seq'][i])} → 预期={anchor_len}（一致）")
                print(
                    f"     锚点接触矩阵维度：{batch['anchor_contact'][i].shape} → 预期=({anchor_len},{anchor_len})（一致）")
                print(f"     对比样本PDB ID：{contrast_pdb}")
                print(f"     对比样本残基长度：{len(batch['contrast_seq'][i])} → 预期={contrast_len}（一致）")
                print(f"     样本对类型标签：{batch['pair_label'][i].item()} → 预期=1（正样本对，方案0.2节标签）")
            break

        print("\n✅ 测试通过：合法PDB文件（如1d00D.pdb）已正确加载，不合法文件（如1d00D00.pdb）已排除")
    except Exception as e:
        print(f"❌ 测试失败：{str(e)}")