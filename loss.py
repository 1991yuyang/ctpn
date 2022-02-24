from torch import nn
import numpy as np
import torch as t
from focalloss import FocalLoss


class LossFunc(nn.Module):

    def __init__(self, lamda1, lamda2, train_side_ref, use_focal_loss, focal_loss_gamma=2, focal_loss_alpha=0.25):
        super(LossFunc, self).__init__()
        if use_focal_loss:
            self.ce = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        else:
            self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.train_side_ref = train_side_ref

    def forward(self, cls_outputs, reg_outputs, side_ref_outputs, cls_targets, reg_targets, side_ref_targets):
        total_loss_ = 0
        cls_loss_ = 0
        reg_loss_ = 0
        side_ref_loss_ = 0
        batch_size = cls_outputs.size()[0]
        for i in range(batch_size):
            cls_output = cls_outputs[i]
            reg_output = reg_outputs[i]
            side_ref_output = side_ref_outputs[i]
            cls_target = cls_targets[i]
            reg_target = reg_targets[i]
            side_ref_target = side_ref_targets[i]
            cls_point_index = np.array(list(cls_target.keys())).astype(np.int)
            reg_point_index = np.array(list(reg_target.keys())).astype(np.int)
            side_ref_index = np.array(list(side_ref_target.keys())).astype(np.int)
            cls_output_pos_prob = cls_output[cls_point_index[..., 1], cls_point_index[..., 0], cls_point_index[..., 2] * 2 + 1].view((-1, 1))
            cls_output_neg_prob = cls_output[cls_point_index[..., 1], cls_point_index[..., 0], cls_point_index[..., 2] * 2].view((-1, 1))
            cls_output_ = t.cat([cls_output_neg_prob, cls_output_pos_prob], dim=1)
            cls_target_ = t.tensor(list(cls_target.values())).type(t.LongTensor).cuda(0)
            if reg_point_index.tolist():
                reg_output_vc = reg_output[reg_point_index[..., 1], reg_point_index[..., 0], reg_point_index[..., 2] * 2]
                reg_output_vh = reg_output[reg_point_index[..., 1], reg_point_index[..., 0], reg_point_index[..., 2] * 2 + 1]
                reg_target_vc = t.from_numpy(np.array(list(reg_target.values()))[..., 0]).type(t.FloatTensor).cuda(0)
                reg_target_vh = t.from_numpy(np.array(list(reg_target.values()))[..., 1]).type(t.FloatTensor).cuda(0)
            if side_ref_index.tolist():
                side_ref_output_ = side_ref_output[side_ref_index[..., 1], side_ref_index[..., 0], side_ref_index[..., 2]]
                side_ref_target_ = t.tensor(list(side_ref_target.values())).type(t.FloatTensor).cuda(0)
            cls_loss = self.ce(cls_output_, cls_target_)
            if reg_point_index.tolist():
                reg_loss = self.smooth_l1(t.cat([reg_output_vc, reg_output_vh], dim=0), t.cat([reg_target_vc, reg_target_vh], dim=0))
            else:
                reg_loss = 0
            if side_ref_index.tolist():
                side_ref_loss = self.smooth_l1(side_ref_output_, side_ref_target_)
            else:
                side_ref_loss = 0
            if self.train_side_ref:
                total_loss = cls_loss + self.lamda1 * reg_loss + self.lamda2 * side_ref_loss
            else:
                total_loss = cls_loss + self.lamda1 * reg_loss
            total_loss_ += total_loss
            cls_loss_ += cls_loss
            reg_loss_ += reg_loss
            side_ref_loss_ += side_ref_loss
        return total_loss_ / batch_size, cls_loss_ / batch_size, reg_loss_ / batch_size, side_ref_loss_ / batch_size

