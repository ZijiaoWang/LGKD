from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth


def kd_glocal_loss(logits_student_in, logits_teacher_in, temperature, alpha=1.0):
    batch_size, class_num = logits_teacher_in.shape
    temperature_s = alpha * temperature
    temperature_t = temperature
    pred_student = F.softmax(logits_student_in / temperature_s, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature_t, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    # student_matrix_norm = normalize_matrix(student_matrix)
    # teacher_matrix_norm = normalize_matrix(teacher_matrix)
    consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    return consistency_loss

def kd_local_matrix_loss(logits_student_in, logits_teacher_in, temperature, alpha):
    batch_size, class_num = logits_teacher_in.shape
    temperature_s = alpha * temperature
    temperature_t = temperature
    pred_student = F.softmax(logits_student_in / temperature_s, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature_t, dim=1)
    new_pred_student = pred_student[int(0.25*batch_size):int(0.75*batch_size+1), int(0.25*class_num):int(0.75*class_num+1)]
    new_pred_teacher = pred_teacher[int(0.25*batch_size):int(0.75*batch_size+1), int(0.25*class_num):int(0.75*class_num+1)]
    new_batch_size, _= new_pred_student.shape
    student_matrix = torch.mm(new_pred_student, new_pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(new_pred_teacher, new_pred_teacher.transpose(1, 0))
    # student_matrix_norm = normalize_matrix(student_matrix)
    # teacher_matrix_norm = normalize_matrix(teacher_matrix)
    consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / new_batch_size
    return consistency_loss

#---------------------------------------------------------------------------------------------------
def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, gamma, temperature):
    gt_mask = _get_gt_mask(logits_student_in, target)
    other_mask = _get_other_mask(logits_student_in, target)
    T_student = alpha * temperature
    T_teacher = temperature
    pred_student = F.softmax(logits_student_in / T_student, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / T_teacher, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    # average kl
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (T_teacher**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher_in / T_teacher - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student_in / T_student - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (T_teacher**2)
        / target.shape[0]
    )
    return beta * tckd_loss + gamma * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
#--------------------------------------------------------------------------------------------------

class DKD_Ours(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DKD_Ours, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        # self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.ce_loss_weight = 1.0
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.alpha = cfg.KD_Ours.ALPHA
        self.beta = cfg.KD_Ours.BETA
        self.gamma = cfg.KD_Ours.GAMMA
        self.warmup = 20
        self.kd_hard_weight = 1.0
    
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        pred_teacher = F.softmax(logits_teacher.detach(), dim=1)
        confidence, _ = pred_teacher.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)      
        # loss_kd_local = self.kd_loss_weight * ((kd_loss(
        #     logits_student,
        #     logits_teacher,
        #     self.temperature,
        #     1
        # ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
        #     logits_student,
        #     logits_teacher,
        #     self.temperature,
        #     0.5
        # ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
        #     logits_student,
        #     logits_teacher,
        #     self.temperature,
        #     0.75
        # ) * mask).mean())

        loss_kd_local = min(kwargs["epoch"] / self.warmup, 1.0) * ((dkd_loss(
            logits_student,
            logits_teacher,
            target,
            1,
            self.beta,
            self.gamma,
            self.temperature
        ) * mask).mean()) + min(kwargs["epoch"] / self.warmup, 1.0) * ((dkd_loss(
            logits_student,
            logits_teacher,
            target,
            0.5,
            self.beta,
            self.gamma,
            self.temperature
        ) * mask).mean()) + min(kwargs["epoch"] / self.warmup, 1.0) * ((dkd_loss(
            logits_student,
            logits_teacher,
            target,
            0.75,
            self.beta,
            self.gamma,
            self.temperature
        ) * mask).mean())

        loss_kd_local_matrix = self.kd_hard_weight * ((kd_local_matrix_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            1,
        ) * mask).mean()) + self.kd_hard_weight * ((kd_local_matrix_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            0.5,
        ) * mask).mean()) + self.kd_hard_weight * ((kd_local_matrix_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            0.75,
        ) * mask).mean())

        loss_kd_global = self.kd_hard_weight * ((kd_glocal_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            1,
        ) * mask).mean()) + self.kd_hard_weight * ((kd_glocal_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            0.5,
        ) * mask).mean()) + self.kd_hard_weight * ((kd_glocal_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            0.75,
        ) * mask).mean())

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd_local": loss_kd_local,
            "loss_kd_local_matrix": loss_kd_local_matrix,
            "loss_kd_global": loss_kd_global
        }
        return logits_student, losses_dict
