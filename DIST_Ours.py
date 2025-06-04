from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(pred_student, pred_teacher):
    return 1 - pearson_correlation(pred_student, pred_teacher).mean()


def intra_class_relation(pred_student, pred_teacher):
    return inter_class_relation(pred_student.transpose(0, 1), pred_teacher.transpose(0, 1))


def kd_local_loss(logits_student_in, logits_teacher_in, temperature, alpha=1.0, beta=1.0, gamma=1.0):
    temperature_s = alpha * temperature
    temperature_t = temperature
    pred_student = F.softmax(logits_student_in / temperature_s, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature_t, dim=1)
    inter_loss = temperature**2 * inter_class_relation(pred_student, pred_teacher)
    intra_loss = temperature**2 * intra_class_relation(pred_student, pred_teacher)
    kd_local_loss = beta * inter_loss + gamma * intra_loss
    return kd_local_loss

def kd_glocal_loss(logits_student_in, logits_teacher_in, temperature, alpha=1.0):
    batch_size, class_num = logits_teacher_in.shape
    temperature_s = alpha * temperature
    temperature_t = temperature
    pred_student = F.softmax(logits_student_in / temperature_s, dim=1)
    pred_teacher = F.softmax(logits_teacher_in / temperature_t, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
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
    consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / new_batch_size
    return consistency_loss

class DIST_Ours(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DIST_Ours, self).__init__(student, teacher)
        self.temperature = 4
        self.kd_loss_weight = 1.0
        self.kd_hard_weight = 1.0
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
    
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
        loss_kd_local = self.kd_loss_weight * ((kd_local_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            self.alpha,
            self.beta,
            self.gamma,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_local_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            0.5,
            self.beta,
            self.gamma,
        ) * mask).mean()) + self.kd_loss_weight * ((kd_local_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            0.75,
            self.beta,
            self.gamma,
        ) * mask).mean()) 
        
        loss_kd_local_matrix = self.kd_hard_weight * ((kd_local_matrix_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            self.alpha,
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
            self.alpha,
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
            "loss_kd_local": loss_kd_local,
            "loss_kd_local_matrix": loss_kd_local_matrix,
            "loss_kd_global": loss_kd_global
        }
        return logits_student, losses_dict
