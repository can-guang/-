# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn


def My_KL_loss(predictions, true_distributions):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum()
    KL = -1.0 * KL / predictions.shape[0]
    return KL


def My_logit_ML_loss(view_predictions, true_labels):
    view_predictions_sig = torch.sigmoid(view_predictions)
    criterion = nn.BCELoss()
    # print(view_predictions_sig)
    ML_loss = criterion(view_predictions_sig, true_labels)
    print(ML_loss)
    return ML_loss

def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    print(loss)
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

