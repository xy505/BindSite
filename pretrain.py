import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from model import (
    MODEL_CONFIG,
    ProtT5Embedding,
    ContactGuidedAugment
)
from pre_dataloader import (
    ToughM1PairDataset,
    toughM1_pair_collate_fn,
    CONFIG as DATALOADER_CONFIG
)

# --------------------------
# 1. 预训练配置
# --------------------------
PRETRAIN_CONFIG = {
    "epochs": 15,
    "lr": MODEL_CONFIG.get("lr", 2e-4),
    "weight_decay": MODEL_CONFIG.get("weight_decay", 1e-5),
    "margin": 1.0,
    "log_interval": 5,
    "checkpoint_dir": "./pretrain_checkpoints",
    "best_model_name": "best_emb_augment1.pth",
    "resume_pretrain": True,
    "resume_model_name": "latest_emb_augment1.pth",
    "val_sample_size": 200  # 验证集随机样本数
}

os.makedirs(PRETRAIN_CONFIG["checkpoint_dir"], exist_ok=True)


# --------------------------
# 2. 二元对比损失（保持不变）
# --------------------------
class BinaryContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor_feat, contrast_feat, pair_label):
        similarity = self.cos_sim(anchor_feat, contrast_feat)
        distance = 1 - similarity

        pos_loss = pair_label * torch.pow(distance, 2)
        neg_loss = (1 - pair_label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        total_loss = (pos_loss + neg_loss).mean()
        return total_loss, {"similarity": similarity, "distance": distance}


# --------------------------
# 3. 预训练模型（保持不变）
# --------------------------
class ContrastivePretrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.protT5 = ProtT5Embedding()
        for param in self.protT5.parameters():
            param.requires_grad = False
        self.protT5.eval()

        self.emb_augment = ContactGuidedAugment().to(MODEL_CONFIG["device"])
        self.global_pool = self._binding_site_avg_pool
        self.contrast_loss = BinaryContrastiveLoss(margin=PRETRAIN_CONFIG["margin"])

    def _binding_site_avg_pool(self, aug_emb, bs_label):
        bs_mask = bs_label.bool()
        bs_feats = aug_emb[bs_mask]

        if len(bs_feats) == 0:
            return torch.mean(aug_emb, dim=0)
        return torch.mean(bs_feats, dim=0)

    def _extract_pair_feats(self, batch):
        batch_size = len(batch["anchor_seq"])
        anchor_feats = []
        contrast_feats = []

        for i in range(batch_size):
            with torch.no_grad():
                anchor_protT5 = self.protT5(
                    residue_seqs=[batch["anchor_seq"][i]],
                    seq_lens=[batch["anchor_seq_len"][i].item()]
                )[0].to(MODEL_CONFIG["device"])
                contrast_protT5 = self.protT5(
                    residue_seqs=[batch["contrast_seq"][i]],
                    seq_lens=[batch["contrast_seq_len"][i].item()]
                )[0].to(MODEL_CONFIG["device"])

            anchor_aug = self.emb_augment(
                protT5_emb=anchor_protT5,
                contact_matrix=batch["anchor_contact"][i].to(MODEL_CONFIG["device"])
            )
            contrast_aug = self.emb_augment(
                protT5_emb=contrast_protT5,
                contact_matrix=batch["contrast_contact"][i].to(MODEL_CONFIG["device"])
            )

            anchor_global = self.global_pool(
                aug_emb=anchor_aug,
                bs_label=batch["anchor_bs_label"][i].to(MODEL_CONFIG["device"])
            ).unsqueeze(0)
            contrast_global = self.global_pool(
                aug_emb=contrast_aug,
                bs_label=batch["contrast_bs_label"][i].to(MODEL_CONFIG["device"])
            ).unsqueeze(0)

            anchor_feats.append(anchor_global)
            contrast_feats.append(contrast_global)

        return (
            torch.cat(anchor_feats, dim=0),
            torch.cat(contrast_feats, dim=0),
            batch["pair_label"].to(MODEL_CONFIG["device"])
        )

    def forward(self, batch):
        anchor_feat, contrast_feat, pair_label = self._extract_pair_feats(batch)
        loss, metrics = self.contrast_loss(anchor_feat, contrast_feat, pair_label)

        return loss, {
            "anchor_feats": anchor_feat,
            "contrast_feats": contrast_feat,
            "similarity": metrics["similarity"],
            "pair_label": pair_label,
            "anchor_pdb": batch["anchor_pdb"],
            "contrast_pdb": batch["contrast_pdb"]
        }


# --------------------------
# 4. 断点加载/保存函数（保持不变）
# --------------------------
def load_pretrain_checkpoint(model, optimizer, scheduler):
    resume_path = os.path.join(
        PRETRAIN_CONFIG["checkpoint_dir"],
        PRETRAIN_CONFIG["resume_model_name"]
    )
    if not os.path.exists(resume_path) or not PRETRAIN_CONFIG["resume_pretrain"]:
        print(f"⚠️  未找到预训练断点或未开启续训，将从头开始预训练")
        return model, optimizer, scheduler, 1, float("inf")

    checkpoint = torch.load(resume_path, map_location=MODEL_CONFIG["device"])
    model.emb_augment.load_state_dict(checkpoint["emb_augment_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    resume_epoch = checkpoint["current_epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    print(f"✅ 加载预训练断点：{resume_path}")
    print(f"   - 恢复至第{resume_epoch}轮（上一轮epoch：{checkpoint['current_epoch']}）")
    print(f"   - 上一轮验证损失：{checkpoint['last_val_loss']:.4f}")
    print(f"   - 当前最优验证损失：{best_val_loss:.4f}")
    return model, optimizer, scheduler, resume_epoch, best_val_loss


def save_pretrain_checkpoint(model, optimizer, scheduler, current_epoch, last_val_loss, best_val_loss):
    latest_path = os.path.join(
        PRETRAIN_CONFIG["checkpoint_dir"],
        PRETRAIN_CONFIG["resume_model_name"]
    )
    torch.save({
        "emb_augment_state_dict": model.emb_augment.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "current_epoch": current_epoch,
        "last_val_loss": last_val_loss,
        "best_val_loss": best_val_loss,
        "config": PRETRAIN_CONFIG
    }, latest_path)
    print(f"💾 保存第{current_epoch}轮断点至：{latest_path}")

    if last_val_loss < best_val_loss:
        best_path = os.path.join(
            PRETRAIN_CONFIG["checkpoint_dir"],
            PRETRAIN_CONFIG["best_model_name"]
        )
        torch.save(model.emb_augment.state_dict(), best_path)
        print(f"🏆 保存最优预训练模型至：{best_path}（验证损失：{last_val_loss:.4f}）")
        return last_val_loss
    return best_val_loss


# --------------------------
# 5. 训练/验证函数（保持不变）
# --------------------------
def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_samples = len(train_loader.dataset)
    pos_sims = []
    neg_sims = []

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch}/{PRETRAIN_CONFIG['epochs']}] (Train)"
    )

    for batch in progress_bar:
        optimizer.zero_grad()
        loss, metrics = model(batch)
        loss.backward()
        optimizer.step()

        batch_size = len(batch["anchor_seq"])
        total_loss += loss.item() * batch_size

        sims = metrics["similarity"].detach().cpu().numpy()
        labels = metrics["pair_label"].detach().cpu().numpy()
        pos_sims.extend(sims[labels == 1])
        neg_sims.extend(sims[labels == 0])

        avg_loss_so_far = total_loss / ((progress_bar.n + 1) * batch_size)
        avg_pos_sim = np.mean(pos_sims) if pos_sims else 0.0
        avg_neg_sim = np.mean(neg_sims) if neg_sims else 0.0
        progress_bar.set_postfix({
            "Avg Loss": f"{avg_loss_so_far:.4f}",
            "Pos Sim": f"{avg_pos_sim:.4f}",
            "Neg Sim": f"{avg_neg_sim:.4f}"
        })

    avg_train_loss = total_loss / total_samples
    avg_pos_sim = np.mean(pos_sims) if pos_sims else 0.0
    avg_neg_sim = np.mean(neg_sims) if neg_sims else 0.0
    print(f"\n✅ Epoch [{epoch}] 训练完成：")
    print(f"   - 平均训练损失：{avg_train_loss:.4f}")
    print(f"   - 正样本对结合位点平均相似度：{avg_pos_sim:.4f}")
    print(f"   - 负样本对结合位点平均相似度：{avg_neg_sim:.4f}")
    return avg_train_loss


def validate(model, val_loader, device, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = len(val_loader.dataset)
    pos_sims = []
    neg_sims = []

    with torch.no_grad(), tqdm(
            val_loader,
            desc=f"Epoch [{epoch}/{PRETRAIN_CONFIG['epochs']}] (Val)"
    ) as progress_bar:
        for batch in progress_bar:
            loss, metrics = model(batch)
            batch_size = len(batch["anchor_seq"])
            total_loss += loss.item() * batch_size

            sims = metrics["similarity"].cpu().numpy()
            labels = metrics["pair_label"].cpu().numpy()
            pos_sims.extend(sims[labels == 1])
            neg_sims.extend(sims[labels == 0])

            avg_val_loss_so_far = total_loss / ((progress_bar.n + 1) * batch_size)
            progress_bar.set_postfix({"Avg Val Loss": f"{avg_val_loss_so_far:.4f}"})

    avg_val_loss = total_loss / total_samples
    avg_pos_sim = np.mean(pos_sims) if pos_sims else 0.0
    avg_neg_sim = np.mean(neg_sims) if neg_sims else 0.0
    print(f"\n📊 Epoch [{epoch}] 验证完成：")
    print(f"   - 平均验证损失：{avg_val_loss:.4f}")
    print(f"   - 正样本对结合位点平均相似度：{avg_pos_sim:.4f}")
    print(f"   - 负样本对结合位点平均相似度：{avg_neg_sim:.4f}")
    return avg_val_loss


# --------------------------
# 6. 动态样本选择工具类（核心修改）
# --------------------------
class DynamicSampleSelector:
    @staticmethod
    def select_random_pdb1_samples(dataset):
        """对样本按pdb1分组，每个pdb1随机保留1个样本（适用于正负样本）"""
        # 按pdb1分组所有样本
        pdb1_groups = {}
        for idx in range(len(dataset)):
            # 获取原始数据集中的样本索引
            original_idx = dataset.selected_indices[idx]
            # 获取pdb1标识
            pair_data = dataset.csv_data.iloc[original_idx]
            pdb1 = pair_data["pdb1"]

            if pdb1 not in pdb1_groups:
                pdb1_groups[pdb1] = []
            pdb1_groups[pdb1].append(idx)

        # 每个pdb1随机选择一个样本索引
        selected_indices = []
        for pdb1, indices in pdb1_groups.items():
            selected_indices.append(np.random.choice(indices))

        # 创建临时数据集包装器
        class TempDataset(Dataset):
            def __init__(self, base_dataset, indices):
                self.base = base_dataset
                self.indices = indices

            def __getitem__(self, idx):
                return self.base[self.indices[idx]]

            def __len__(self):
                return len(self.indices)

        return TempDataset(dataset, selected_indices)

    @staticmethod
    def select_random_val_samples(dataset, sample_size):
        """从验证集随机选择指定数量样本"""
        if len(dataset) <= sample_size:
            return dataset

        selected_indices = np.random.choice(
            len(dataset),
            size=sample_size,
            replace=False
        )

        class TempDataset(Dataset):
            def __init__(self, base_dataset, indices):
                self.base = base_dataset
                self.indices = indices

            def __getitem__(self, idx):
                return self.base[self.indices[idx]]

            def __len__(self):
                return len(self.indices)

        return TempDataset(dataset, selected_indices)


# --------------------------
# 7. 预训练主函数（核心修改）
# --------------------------
def main():
    device = MODEL_CONFIG["device"]
    print(f"=== 结合位点方案预训练流程 ===")
    print(f"📌 硬件配置：{device}")
    print(
        f"📌 核心参数：批次大小={DATALOADER_CONFIG['train']['batch_size']}, "
        f"总轮次={PRETRAIN_CONFIG['epochs']}, "
        f"学习率={PRETRAIN_CONFIG['lr']}, "
        f"验证集样本数={PRETRAIN_CONFIG['val_sample_size']}"
    )

    # 初始化基础数据集
    print(f"\n📌 初始化基础数据集...")
    # 训练集（正样本和负样本分开）
    train_pos_dataset = ToughM1PairDataset(split="train", pair_type="positive")
    train_neg_dataset = ToughM1PairDataset(split="train", pair_type="negative")

    # 验证集
    val_pos_dataset = ToughM1PairDataset(split="val", pair_type="positive")
    val_neg_dataset = ToughM1PairDataset(split="val", pair_type="negative")
    full_val_dataset = ConcatDataset([val_pos_dataset, val_neg_dataset])

    # 打印原始数据统计
    print(f"✅ 原始数据集统计：")
    print(f"   - 训练集正样本：{len(train_pos_dataset)}，负样本：{len(train_neg_dataset)}")
    print(f"   - 验证集总样本：{len(full_val_dataset)}（将随机选择{PRETRAIN_CONFIG['val_sample_size']}个）")

    # 初始化模型、优化器、调度器
    print(f"\n📌 初始化预训练模型...")
    model = ContrastivePretrainer().to(device)
    optimizer = AdamW(
        model.emb_augment.parameters(),
        lr=PRETRAIN_CONFIG["lr"],
        weight_decay=PRETRAIN_CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_CONFIG["epochs"]
    )

    # 加载断点
    start_epoch, best_val_loss = 1, float("inf")
    if PRETRAIN_CONFIG["resume_pretrain"]:
        model, optimizer, scheduler, start_epoch, best_val_loss = load_pretrain_checkpoint(model, optimizer, scheduler)

    # 预训练循环（每个epoch动态生成训练集）
    print(f"\n📌 从第{start_epoch}轮开始预训练...")
    for epoch in range(start_epoch, PRETRAIN_CONFIG["epochs"] + 1):
        # 动态处理训练集：正负样本都按pdb1去重，每个pdb1保留1个随机样本
        print(f"\n📌 Epoch {epoch} 训练集处理：")
        # 正样本处理
        unique_pos_dataset = DynamicSampleSelector.select_random_pdb1_samples(train_pos_dataset)
        # 负样本处理
        unique_neg_dataset = DynamicSampleSelector.select_random_pdb1_samples(train_neg_dataset)
        print(f"   - 正样本去重后：{len(unique_pos_dataset)}（原始：{len(train_pos_dataset)}）")
        print(f"   - 负样本去重后：{len(unique_neg_dataset)}（原始：{len(train_neg_dataset)}）")

        # 合并处理后的正样本和负样本
        train_dataset = ConcatDataset([unique_pos_dataset, unique_neg_dataset])
        train_loader = DataLoader(
            train_dataset,
            batch_size=DATALOADER_CONFIG["train"]["batch_size"],
            shuffle=True,
            num_workers=DATALOADER_CONFIG["train"]["num_workers"],
            pin_memory=True,
            collate_fn=toughM1_pair_collate_fn
        )

        # 动态处理验证集：随机选择指定数量样本
        selected_val_dataset = DynamicSampleSelector.select_random_val_samples(
            full_val_dataset,
            PRETRAIN_CONFIG["val_sample_size"]
        )
        val_loader = DataLoader(
            selected_val_dataset,
            batch_size=DATALOADER_CONFIG["train"]["batch_size"],
            shuffle=False,
            num_workers=DATALOADER_CONFIG["train"]["num_workers"],
            pin_memory=True,
            collate_fn=toughM1_pair_collate_fn
        )
        print(f"   - 验证集随机选择：{len(selected_val_dataset)}个样本")

        # 训练和验证
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device, epoch)
        scheduler.step()

        # 保存模型
        best_val_loss = save_pretrain_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss)

        if epoch % PRETRAIN_CONFIG["log_interval"] == 0:
            print(f"\n📊 Epoch [{epoch}/{PRETRAIN_CONFIG['epochs']}] 关键日志：")
            print(f"   - 训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f}")
            print(f"   - 当前最优验证损失：{best_val_loss:.4f}")
            print(f"   - 当前学习率：{optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n=== 预训练完成 ===")
    print(f"📌 核心结果：")
    print(f"   - 最优验证损失：{best_val_loss:.4f}")
    print(
        f"   - 预训练嵌入增强模块路径：{os.path.join(PRETRAIN_CONFIG['checkpoint_dir'], PRETRAIN_CONFIG['best_model_name'])}")


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("⚠️  缺少tqdm，自动安装...")
        os.system("pip install tqdm")
        from tqdm import tqdm

    main()