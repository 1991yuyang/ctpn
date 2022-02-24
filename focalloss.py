import torch as t
from torch import nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        output = self.softmax(output.view((-1, 2)))
        target = target.view(-1)
        positive_index = target == 1
        negtive_index = target == 0
        pos_prob_pred = output[positive_index, 1]
        neg_prob_pred = output[negtive_index, 0]
        pos_loss = t.sum(-self.alpha * t.pow((1.0 - pos_prob_pred), self.gamma) * t.log(pos_prob_pred))
        neg_loss = t.sum(-(1.0 - self.alpha) * t.pow((1.0 - neg_prob_pred), self.gamma) * t.log(neg_prob_pred))
        total_loss = pos_loss + neg_loss
        avg_loss = total_loss / output.size()[0]
        return avg_loss


if __name__ == "__main__":
    loss = FocalLoss()
    d = t.randn(3, 2)
    target = t.tensor([0, 1, 0])
    l = loss(d, target)