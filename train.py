import os
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc  # 新增PR曲线相关导入
from tqdm import tqdm
from dataloader import get_binding_site_dataloader, CONFIG as DATALOADER_CONFIG
from model import BindingSitePredictor, MODEL_CONFIG
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 训练配置（含早停参数）
# --------------------------
TRAIN_CONFIG = {
    "epochs": 15,  # 训练总轮次
    "lr": MODEL_CONFIG["lr"],  # 初始学习率：2e-4
    "weight_decay": MODEL_CONFIG["weight_decay"],  # 权重衰减：1e-5
    "checkpoint_dir": "./bs_checkpoints",  # 模型保存路径
    "log_interval": 5,  # 日志打印间隔
    "best_metric": "f1",  # 最优模型判定标准：F1值
    "threshold": MODEL_CONFIG["threshold"],  # 预测阈值：0.5
    "progress_bar_desc": "Training Batch",  # 进度条描述
    "resume_train": True,  # 开启断点续训
    "resume_checkpoint": "latest_model3-7.pth",  # 断点模型文件名S
    "ablation_type": "None",  # 消融实验类型
    "use_pretrain": False,  # 使用预训练参数
    "pretrain_checkpoint_dir": "./pretrain_checkpoints",  # 预训练模型目录
    "pretrain_checkpoint_name": "latest_emb_augment.pth",  # 预训练断点文件名
    # 早停策略参数
    "early_stopping_patience": 5,  # 容忍无提升的轮次
    "early_stopping_min_delta": 5e-3  # 最小提升阈值
}

# 确保保存目录存在
os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(TRAIN_CONFIG["pretrain_checkpoint_dir"], exist_ok=True)


# --------------------------
# 2. 断点加载函数（兼容旧模型）
# --------------------------
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载断点，兼容无early_stop_counter的旧模型"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  未找到断点文件 {checkpoint_path}，将从头开始训练")
        return model, optimizer, scheduler, 1, 0.0, 0  # 初始化早停计数器为0

    # 加载断点数据
    checkpoint = torch.load(checkpoint_path, map_location=MODEL_CONFIG["device"])

    # 兼容处理：检查是否有early_stop_counter字段
    if "early_stop_counter" in checkpoint:
        early_stop_counter = checkpoint["early_stop_counter"]
    else:
        early_stop_counter = 0  # 旧模型默认初始化为0
        print(f"⚠️  检测到旧版本断点（无早停计数器），自动初始化早停计数器为0")

    # 恢复模型状态
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    resume_epoch = checkpoint["current_epoch"] + 1
    best_metric = checkpoint["best_metric"]

    # 打印恢复信息
    print(f"✅ 成功加载断点文件：{checkpoint_path}")
    print(f"   - 恢复训练：从第 {resume_epoch} 轮开始（上一轮为第 {checkpoint['current_epoch']} 轮）")
    print(f"   - 历史最佳验证{TRAIN_CONFIG['best_metric'].upper()}：{best_metric:.4f}")
    print(f"   - 早停计数器状态：已连续 {early_stop_counter} 轮无提升")
    print(f"   - 上一轮训练损失：{checkpoint['last_train_loss']:.4f}")
    print(f"   - 上一轮验证损失：{checkpoint['last_val_loss']:.4f}")

    return model, optimizer, scheduler, resume_epoch, best_metric, early_stop_counter


# --------------------------
# 3. 断点保存函数（含早停计数器）
# --------------------------
def save_checkpoint(model, optimizer, scheduler, current_epoch, last_train_loss, last_val_loss, best_metric,
                    early_stop_counter):
    """保存断点（包含早停计数器）"""
    checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], TRAIN_CONFIG["resume_checkpoint"])
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "current_epoch": current_epoch,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "best_metric": best_metric,
        "early_stop_counter": early_stop_counter,  # 保存早停计数器
        "config": TRAIN_CONFIG
    }, checkpoint_path)
    print(f"💾 已保存第 {current_epoch} 轮断点至：{checkpoint_path}（含早停计数器）")


# --------------------------
# 4. 指标计算函数
# --------------------------
def calculate_metrics(pred_list, target_list):
    """计算评估指标"""
    pred_flat = []
    target_flat = []
    for pred, target in zip(pred_list, target_list):
        pred_prob = pred.squeeze(-1).cpu().detach().numpy()
        true_label = target.cpu().numpy()
        valid_mask = np.isfinite(pred_prob) & np.isfinite(true_label)
        pred_flat.extend(pred_prob[valid_mask])
        target_flat.extend(true_label[valid_mask])

    pred_flat = np.array(pred_flat)
    target_flat = np.array(target_flat)
    pred_binary = (pred_flat >= TRAIN_CONFIG["threshold"]).astype(int)
    target_binary = target_flat.astype(int)

    # 计算基础统计量
    # TP = np.sum((pred_binary == 1) & (target_binary == 1))
    # TN = np.sum((pred_binary == 0) & (target_binary == 0))
    # FP = np.sum((pred_binary == 1) & (target_binary == 0))
    # FN = np.sum((pred_binary == 0) & (target_binary == 1))
    TP = np.sum((pred_binary == 1) & (target_binary == 1))  # 真阳性（结合位点预测正确）
    TN = np.sum((pred_binary == 0) & (target_binary == 0))  # 真阴性（非结合位点预测正确）
    FP = np.sum((pred_binary == 1) & (target_binary == 0))  # 假阳性（非结合位点误判为结合位点）
    FN = np.sum((pred_binary == 0) & (target_binary == 1))  # 假阴性（结合位点误判为非结合位点）
    # 计算指标
    metrics = {}
    metrics["sen"] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    metrics["spe"] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    total = TP + TN + FP + FN
    metrics["accuracy"] = (TP + TN) / total if total > 0 else 0.0
    metrics["pre"] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    denominator = np.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))
    metrics["mcc"] = (TP * TN - FP * FN) / denominator if denominator != 0 else 0.0
    # denominator = np.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))
    # metrics["mcc"] = (TP * TN - FP * FN) / denominator if denominator != 0 else 0.0
    metrics["f1"] = f1_score(target_binary, pred_binary, zero_division=0)
    metrics["auc"] = roc_auc_score(target_flat, pred_flat) if len(np.unique(target_flat)) == 2 else 0.5

    # 新增：计算AUC-PRC（PR曲线下面积）
    if len(np.unique(target_flat)) == 2:
        precision, recall, _ = precision_recall_curve(target_flat, pred_flat)  # 计算PR曲线
        metrics["auc_prc"] = auc(recall, precision)  # 计算PR曲线下面积
    else:
        metrics["auc_prc"] = 0.5  # 仅一类标签时为随机水平

    return {k: round(v, 4) for k, v in metrics.items()}

    # return {k: round(v, 4) for k, v in metrics.items()}


# --------------------------
# 5. 单轮训练函数
# --------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_samples = len(dataloader.dataset)
    batch_num = len(dataloader)

    with tqdm(
            enumerate(dataloader),
            total=batch_num,
            desc=f"Epoch [{epoch}/{TRAIN_CONFIG['epochs']}] | {TRAIN_CONFIG['progress_bar_desc']}",
            unit="batch",
            leave=True
    ) as pbar:
        for batch_idx, batch in pbar:
            current_batch_size = len(batch["residue_sequence"])
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * current_batch_size
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss (so far)": f"{avg_loss_so_far:.4f}"
            })

    epoch_avg_loss = total_loss / total_samples
    print(f"\n✅ Epoch [{epoch}/{TRAIN_CONFIG['epochs']}] 训练完成：")
    print(f"   - 本轮平均损失：{epoch_avg_loss:.4f}")
    return epoch_avg_loss


# --------------------------
# 6. 单轮验证/测试函数
# --------------------------
def evaluate(model, dataloader, device, split="val"):
    model.eval()
    total_loss = 0.0
    total_samples = len(dataloader.dataset)
    all_preds = []
    all_targets = []

    with torch.no_grad(), tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Evaluating [{split.upper()}] Set",
            unit="batch",
            leave=True
    ) as pbar:
        for batch in pbar:
            loss, pred_list = model(batch)
            current_batch_size = len(batch["residue_sequence"])
            total_loss += loss.item() * current_batch_size

            all_preds.extend(pred_list)
            all_targets.extend(batch["binding_site_label"])
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_samples
    metrics = calculate_metrics(all_preds, all_targets)

    print(f"\n[{split.upper()}] 评估结果：")
    print(f"   - 平均损失：{avg_loss:.4f}")
    print(f"   - 灵敏度（Sen）：{metrics['sen']:.4f}")
    print(f"   - 特异度（Spe）：{metrics['spe']:.4f}")
    print(f"   - 准确率（Accuracy）：{metrics['accuracy']:.4f}")
    print(f"   - 精确率（Pre）：{metrics['pre']:.4f}")
    print(f"   - 马修斯相关系数（MCC）：{metrics['mcc']:.4f}")
    print(f"   - F1值：{metrics['f1']:.4f}")
    print(f"   - AUC-ROC：{metrics['auc']:.4f}")
    print(f"   - AUC-PRC：{metrics['auc_prc']:.4f}")

    return avg_loss, metrics


# --------------------------
# 7. 消融实验+预训练参数控制
# --------------------------
def set_ablation_mode(model, ablation_type=None, use_pretrain=TRAIN_CONFIG["use_pretrain"]):
    """加载预训练参数或设置消融模式"""

    def _inner_load_pretrain_emb(resume_path):
        if not os.path.exists(resume_path):
            return model, False
        checkpoint = torch.load(resume_path, map_location=MODEL_CONFIG["device"])
        model.emb_augment.load_state_dict(checkpoint["emb_augment_state_dict"])
        print(f"✅ 加载预训练参数：{resume_path}")
        # print(f"   - 对应预训练轮次：第{checkpoint['current_epoch']}轮")
        return model, True

    if use_pretrain:
        pretrain_resume_path = os.path.join(
            TRAIN_CONFIG["pretrain_checkpoint_dir"],
            TRAIN_CONFIG["pretrain_checkpoint_name"]
        )
        model, load_success = _inner_load_pretrain_emb(pretrain_resume_path)
        if load_success:
            for param in model.emb_augment.parameters():
                param.requires_grad = True
        else:
            print(f"⚠️  未找到预训练文件，emb_augment随机初始化")
    else:
        print(f"✅ 不使用预训练参数，emb_augment随机初始化")
        model.emb_augment.W_contact.requires_grad = True

    if ablation_type is None or ablation_type == "None":
        model.dynamic_gcn.num_iter = MODEL_CONFIG["num_dynamic_iter"]
        model.emb_augment.num_heads = MODEL_CONFIG["num_heads"]
        return model

    if ablation_type == "w/o WA":
        model.emb_augment.W_contact.data = torch.zeros_like(model.emb_augment.W_contact.data)
        model.emb_augment.W_contact.requires_grad = False
        print(f"✅ 消融模式：{ablation_type}")
    elif ablation_type == "w/o DM":
        model.dynamic_gcn.num_iter = 1
        print(f"✅ 消融模式：{ablation_type}")
    elif ablation_type == "w/o ML":
        model.emb_augment.num_heads = 1
        model.emb_augment.head_dim = MODEL_CONFIG["node_dim"]
        print(f"✅ 消融模式：{ablation_type}")
    else:
        raise ValueError(f"消融类型错误：仅支持'None'/'w/o WA'/'w/o DM'/'w/o ML'")

    return model


# --------------------------
# 8. 主训练函数（含早停逻辑）
# --------------------------
def main(ablation_type=TRAIN_CONFIG["ablation_type"]):
    # 初始化设备
    device = MODEL_CONFIG["device"]
    print(f"=== 结合位点预测模型训练（含早停策略） ===")
    print(f"📌 硬件配置：{device}")
    print(f"📌 早停策略：当F1指标连续 {TRAIN_CONFIG['early_stopping_patience']} 轮无提升时停止训练")
    print(f"📌 最小提升阈值：{TRAIN_CONFIG['early_stopping_min_delta']}")

    # 加载数据集
    print(f"\n📌 加载数据集...")
    train_loader = get_binding_site_dataloader(split="train")
    val_loader = get_binding_site_dataloader(split="val")
    test_loader = get_binding_site_dataloader(split="test")
    print(f"✅ 数据集加载完成：")
    print(f"   - 训练集：{len(train_loader.dataset)}样本，{len(train_loader)}批次")
    print(f"   - 验证集：{len(val_loader.dataset)}样本，{len(val_loader)}批次")
    print(f"   - 测试集：{len(test_loader.dataset)}样本，{len(test_loader)}批次")

    # 初始化模型、优化器、调度器
    print(f"\n📌 初始化模型与优化器...")
    model = BindingSitePredictor().to(device)
    model = set_ablation_mode(model, ablation_type)
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])

    # 断点恢复（兼容旧模型）
    start_epoch = 1
    best_metric = 0.0
    early_stop_counter = 0  # 早停计数器
    if TRAIN_CONFIG["resume_train"]:
        checkpoint_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], TRAIN_CONFIG["resume_checkpoint"])
        model, optimizer, scheduler, start_epoch, best_metric, early_stop_counter = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path
        )

    # 训练循环（含早停逻辑）
    print(f"\n📌 开始训练...")
    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"] + 1):
        # 单轮训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        # 单轮验证
        val_loss, val_metrics = evaluate(model, val_loader, device, split="val")
        # 学习率更新
        scheduler.step()

        # 早停逻辑判断
        current_metric = val_metrics[TRAIN_CONFIG["best_metric"]]
        metric_improvement = current_metric - best_metric

        if metric_improvement > TRAIN_CONFIG["early_stopping_min_delta"]:
            # 指标提升：重置计数器，更新最佳模型
            early_stop_counter = 0
            best_metric = current_metric
            ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
            best_model_filename = f"{MODEL_CONFIG['best_model_name'].rsplit('.', 1)[0]}{ablation_suffix}.pth"
            best_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], best_model_filename)
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 保存最优模型（Val {TRAIN_CONFIG['best_metric'].upper()}: {best_metric:.4f}）")
        else:
            # 指标无提升：计数器+1
            early_stop_counter += 1
            print(f"⏸️  早停监控：F1指标连续 {early_stop_counter}/{TRAIN_CONFIG['early_stopping_patience']} 轮无提升")

        # 保存断点（包含早停计数器）
        save_checkpoint(
            model, optimizer, scheduler,
            current_epoch=epoch,
            last_train_loss=train_loss,
            last_val_loss=val_loss,
            best_metric=best_metric,
            early_stop_counter=early_stop_counter
        )

        # 日志打印
        if epoch % TRAIN_CONFIG["log_interval"] == 0 or epoch == 1:
            print(f"\n📊 Epoch [{epoch}/{TRAIN_CONFIG['epochs']}] 关键日志：")
            print(f"   - 训练集平均损失：{train_loss:.4f}")
            print(f"   - 验证集平均损失：{val_loss:.4f}")
            print(f"   - 验证集核心指标：F1={val_metrics['f1']:.4f}，最佳F1={best_metric:.4f}")

        # 检查早停条件
        if early_stop_counter >= TRAIN_CONFIG["early_stopping_patience"]:
            print(f"\n🛑 触发早停：F1指标已连续 {TRAIN_CONFIG['early_stopping_patience']} 轮无提升")
            print(f"   - 最佳验证F1值：{best_metric:.4f}")
            break

    # 测试最优模型
    print(f"\n📌 测试最优模型...")
    ablation_suffix = f"_{ablation_type}" if ablation_type is not None else ""
    best_model_filename = f"{MODEL_CONFIG['best_model_name'].rsplit('.', 1)[0]}{ablation_suffix}.pth"
    best_model_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], best_model_filename)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✅ 加载最优模型：{best_model_path}")
    else:
        print(f"⚠️  未找到最优模型，使用最后一轮模型测试")

    test_loss, test_metrics = evaluate(model, test_loader, device, split="test")

    # 打印最终结果
    print(f"\n=== 训练完成 ===")
    print(f"📋 测试集最终指标：")
    print(f"   - F1值：{test_metrics['f1']:.4f}")
    print(f"   - 灵敏度：{test_metrics['sen']:.4f}，特异度：{test_metrics['spe']:.4f}")
    print(f"   - AUC-PRC：{test_metrics['auc_prc']:.4f}")
    print(f"   - 测试集平均损失：{test_loss:.4f}")


# --------------------------
# 执行训练
# --------------------------
if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("⚠️  安装tqdm...")
        os.system("pip install tqdm")
        from tqdm import tqdm

    main(ablation_type=None)