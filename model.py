import torch as t
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):

    def __init__(self, backbone_type):
        super(FeatureExtractor, self).__init__()
        self.unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)
        if backbone_type == "resnet18":
            self.spatial_feature_extractor = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-3])
        if backbone_type == "resnet34":
            self.spatial_feature_extractor = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-3])
        if backbone_type == "resnet50":
            self.spatial_feature_extractor = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-3])
        if backbone_type == "vgg":
            self.spatial_feature_extractor = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        if backbone_type == "vgg":
            out_channels_ = 512
        else:
            out_channels_ = list(list(self.spatial_feature_extractor.children())[-1].children())[-1].conv1.in_channels
        self.brnn = nn.GRU(out_channels_ * 3 * 3, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.spatial_feature_extractor(x)  # [N, C, H, W]
        N, _, H, W = out.size()
        slide_window_result = self.unfold(out)  # [N, C * 3 * 3, H * W]
        first_reshape_result = slide_window_result.view((slide_window_result.size()[0] * H, W, -1))  # [N * H, W, C * 3 * 3]
        lstm_result, _ = self.brnn(first_reshape_result)
        second_reshape_result = lstm_result.contiguous().view((N, -1, H, W))  # [N, 256, H, W]
        fc_feature = self.fc(second_reshape_result)  # [N, 512, H, W]
        return fc_feature


class CTPN(nn.Module):

    def __init__(self, anchor_count, backbone_type):
        super(CTPN, self).__init__()
        self.feat_extr = FeatureExtractor(backbone_type)
        self.rpn_class = nn.Conv2d(in_channels=512, out_channels=anchor_count * 2, kernel_size=1, stride=1, padding=0)  # 用于预测每个anchor有文字/没有文字的概率
        self.rpn_regress = nn.Conv2d(in_channels=512, out_channels=2 * anchor_count, kernel_size=1, stride=1, padding=0)  # * 2 表示只预测bounding box的(y, h), bounding box的宽度固定为16不需要学习
        self.side_refine = nn.Conv2d(in_channels=512, out_channels=anchor_count, kernel_size=1, stride=1, padding=0)  # 用于文本框边缘调整

    def forward(self, x):
        feature = self.feat_extr(x)
        rpn_cls = self.rpn_class(feature).permute(dims=[0, 2, 3, 1])  # [N, H, W, 2 * anchor_count]
        rpn_reg = self.rpn_regress(feature).permute(dims=[0, 2, 3, 1])  # [N, H, W, anchor_count * 2]
        side_ref = self.side_refine(feature).permute(dims=[0, 2, 3, 1]).contiguous() # [N, H, W, anchor_count]
        return rpn_cls, rpn_reg, side_ref


if __name__ == "__main__":
    d = t.randn(1, 3, 512, 256)
    model = CTPN(anchor_count=10, backbone_type="vgg")
    rpn_cls, rpn_reg, side_ref = model(d)
    # print(rpn_reg.squeeze(0)[[1, 2, 3], [2, 3, 4], [0, 15, 19]].size())
