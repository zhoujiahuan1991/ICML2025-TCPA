import torch
import numpy as np
from timm.data import Mixup
from models.utils import Dataset_N_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for mixup
def get_mixup_fn(args):
    mixup_fn = Mixup(
        mixup_alpha=0.8,  # Mixup 系数，0.0 到 1.0 之间，通常设置为 1.0
        cutmix_alpha=args.cutmix_alpha,  # CutMix 系数，通常设置为 0.0 表示不使用 CutMix
        cutmix_minmax=None,  # CutMix 裁剪比例的最小和最大值，通常设置为 None
        prob=0.5,  # 应用 Mixup 或 CutMix 的概率，通常设置为 0.5
        switch_prob=0.5,  # 在 Mixup 和 CutMix 之间切换的概率，通常设置为 0.5
        mode='batch',  # 混合模式，'batch' 表示批次级别混合，'elem' 表示样本级别混合
        label_smoothing=0.1,  # 标签平滑参数，通常设置为 0.1
        num_classes=Dataset_N_classes[args.dataset]  # 类别数量
        )
    return mixup_fn


# for schedule
def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

