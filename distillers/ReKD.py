import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def rekd_loss(logits_student_in, logits_teacher_in, k, alpha, beta, temperature):
    s_mask = top_k_mask(logits_student_in, k)
    not_s_mask = not_topk_mask(logits_student_in, k)
    pred_teacher_part1 = F.softmax(
        logits_teacher_in / temperature - 1000.0 * not_s_mask, dim=1
    )
    log_pred_student_part1 = F.log_softmax(
        logits_student_in / temperature - 1000.0 * not_s_mask, dim=1
    )
    tisd_loss = (
        F.kl_div(log_pred_student_part1, pred_teacher_part1, reduction='sum')
        * (temperature**2)
        / logits_student_in.shape[0]
    )
    # 如果在softmax函数中对特定类别的logits减去一个非常大的数值（例如1000.0），那么这个类别的logit将变得非常小，导致其在softmax函数中的指数变得非常接近于0。
    # 因此，这个类别的概率将趋近于0
    pred_teacher_part2 = F.softmax(
        logits_teacher_in / temperature - 1000.0 * s_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student_in / temperature - 1000.0 * s_mask, dim=1
    )
    ntid_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / logits_student_in.shape[0]
    )
    return alpha * tisd_loss + beta * ntid_loss

# obtain head categories
# K represents Top-K category labelss
def top_k_mask(logits, k):         
    _, indices = logits.topk(k, dim = 1, largest = True, sorted = True)
    # 创建一个与logits张量形状相同的掩码张量mask
    mask = torch.zeros_like(logits).scatter_(1, indices, 1).bool()
    return mask

#obtain tail category mask
def not_topk_mask(logits, k):
    _, indices = logits.topk(k, dim = 1, largest = True, sorted = True)
    mask = torch.ones_like(logits).scatter_(1, indices, 0).bool()
    return mask


class ReKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(ReKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.REKD.CE_WEIGHT
        self.alpha = cfg.REKD.ALPHA
        self.beta = cfg.REKD.BETA
        self.temperature = cfg.REKD.T
        self.warmup = cfg.REKD.WARMUP
        self.k = cfg.REKD.K 

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_rekd = min(kwargs["epoch"] / self.warmup, 1.0) * rekd_loss(
            logits_student,
            logits_teacher,
            self.k,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_rekd,
        }
        return logits_student, losses_dict