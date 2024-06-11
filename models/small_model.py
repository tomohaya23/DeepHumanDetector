#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn


def autopad(k, p=None):
    """
    Pad the input feature map to ensure the output size remains the same after convolution.
    :param k: kernel size
    :param p: padding size (default: k // 2)
    :return: padding size
    """
    if p is None:
        p = k // 2
    return p


class CBS(nn.Module):
    """
    Convolution, Batch Normalization, and SiLU activation layer.
    """

    def __init__(self, in_ch, out_ch, k_size=1, stride=1, padding=None, groups=1):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride, autopad(k_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELAN(nn.Module):
    """
    Efficient Layer Aggregation Network (ELAN) module.
    """

    def __init__(self, ch1, ch2, ch3, n_stack, ids):
        super(ELAN, self).__init__()
        self.ids = ids
        self.n_stack = n_stack
        self.cv1 = CBS(ch1, ch2, 1, 1)
        self.cv2 = CBS(ch1, ch2, 1, 1)
        self.cv3 = nn.ModuleList([CBS(ch2, ch2, 3, 1) for _ in range(n_stack)])
        self.cv4 = CBS(ch2 * 2 + ch2 * (len(ids) - 2), ch3, 1, 1)

    def forward(self, x):
        output1 = self.cv1(x)
        output2 = self.cv2(x)
        output_all = [output1, output2]

        for i in range(self.n_stack):
            output2 = self.cv3[i](output2)
            output_all.append(output2)

        output = self.cv4(torch.cat([output_all[id] for id in self.ids], dim=1))
        return output


class MP(nn.Module):
    """
    Max Pooling layer.
    """

    def __init__(self, k_size=2, stride=2):
        super(MP, self).__init__()
        self.mp = nn.MaxPool2d(k_size, stride)

    def forward(self, x):
        return self.mp(x)


class Backbone(nn.Module):
    """
    Backbone network.
    """

    def __init__(self, hidden_ch, block_ch, n_stack, ids):
        super(Backbone, self).__init__()
        self.stage1 = CBS(3, hidden_ch * 2, 3, 2)
        self.stage2 = nn.Sequential(
            CBS(hidden_ch * 2, hidden_ch * 4, 3, 2),
            ELAN(hidden_ch * 4, block_ch * 2, hidden_ch * 4, n_stack=n_stack, ids=ids),
        )
        self.stage3 = nn.Sequential(
            MP(),
            ELAN(hidden_ch * 4, block_ch * 4, hidden_ch * 8, n_stack=n_stack, ids=ids),
        )
        self.stage4 = nn.Sequential(
            MP(),
            ELAN(hidden_ch * 8, block_ch * 8, hidden_ch * 16, n_stack=n_stack, ids=ids),
        )
        self.stage5 = nn.Sequential(
            MP(),
            ELAN(hidden_ch * 16, block_ch * 16, hidden_ch * 32, n_stack=n_stack, ids=ids),
        )

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class SPPCSPC(nn.Module):
    """
    SPP (Spatial Pyramid Pooling) and CSPC (Cross-Stage Partial Connections) module.
    """

    def __init__(self, ch1, ch2, k_size_list=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        self.cv1 = CBS(ch1, ch2, 1, 1)
        self.cv2 = CBS(ch1, ch2, 1, 1)
        self.cv3 = CBS(ch2, ch2, 3, 1)
        self.cv4 = CBS(ch2, ch2, 1, 1)
        self.mp = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k_size, stride=1, padding=k_size // 2) for k_size in k_size_list]
        )
        self.cv5 = CBS(4 * ch2, ch2, 1, 1)
        self.cv6 = CBS(ch2, ch2, 3, 1)
        self.cv7 = CBS(2 * ch2, ch2, 1, 1)

    def forward(self, x):
        output1 = self.cv4(self.cv3(self.cv1(x)))
        output2 = self.cv6(self.cv5(torch.cat([output1] + [mp(output1) for mp in self.mp], dim=1)))
        output3 = self.cv2(x)
        output = self.cv7(torch.cat((output2, output3), dim=1))
        return output


class NearestUpsample(nn.Module):
    """
    Nearest neighbor upsampling layer.
    """

    def __init__(self, scale_factor):
        super(NearestUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        new_height, new_width = height * self.scale_factor, width * self.scale_factor

        x_reshaped = x.view(batch_size, channels, height, 1, width, 1)
        x_upsample = x_reshaped.expand(batch_size, channels, height, self.scale_factor, width, self.scale_factor)
        x_upsample = x_upsample.contiguous().view(batch_size, channels, new_height, new_width)

        return x_upsample


class Neck(nn.Module):
    """
    Neck (feature fusion) module.
    """

    def __init__(self, hidden_ch, panet_ch, n_stack, ids):
        super(Neck, self).__init__()
        self.hidden_ch = hidden_ch
        self.sppcspc = SPPCSPC(hidden_ch * 32, hidden_ch * 16)

        self.conv_for_p5 = CBS(hidden_ch * 16, hidden_ch * 8)
        self.conv_for_c4 = CBS(hidden_ch * 16, hidden_ch * 8)
        self.conv_for_upsample1 = ELAN(
            hidden_ch * 16,
            panet_ch * 4,
            hidden_ch * 8,
            n_stack=n_stack,
            ids=ids,
        )

        self.conv_for_p4 = CBS(hidden_ch * 8, hidden_ch * 4)
        self.conv_for_c3 = CBS(hidden_ch * 8, hidden_ch * 4)
        self.conv_for_upsample2 = ELAN(
            hidden_ch * 8,
            panet_ch * 2,
            hidden_ch * 4,
            n_stack=n_stack,
            ids=ids,
        )

        self.down_sample1 = CBS(hidden_ch * 4, hidden_ch * 8, k_size=3, stride=2)
        self.conv_for_downsample1 = ELAN(
            hidden_ch * 16,
            panet_ch * 4,
            hidden_ch * 8,
            n_stack=n_stack,
            ids=ids,
        )

        self.down_sample2 = CBS(hidden_ch * 8, hidden_ch * 16, k_size=3, stride=2)
        self.conv_for_downsample2 = ELAN(
            hidden_ch * 32,
            panet_ch * 8,
            hidden_ch * 16,
            n_stack=n_stack,
            ids=ids,
        )

        self.upsample = NearestUpsample(scale_factor=2)

    def forward(self, c3, c4, c5):
        p5 = self.sppcspc(c5)
        p5_conv = self.conv_for_p5(p5)
        p5_upsample = self.upsample(p5_conv)
        p4 = torch.cat([self.conv_for_c4(c4), p5_upsample], dim=1)
        p4 = self.conv_for_upsample1(p4)
        p4_conv = self.conv_for_p4(p4)
        p4_upsample = self.upsample(p4_conv)
        p3 = torch.cat([self.conv_for_c3(c3), p4_upsample], dim=1)
        p3 = self.conv_for_upsample2(p3)
        p3_downsample = self.down_sample1(p3)
        p4 = torch.cat([p3_downsample, p4], dim=1)
        p4 = self.conv_for_downsample1(p4)
        p4_downsample = self.down_sample2(p4)
        p5 = torch.cat([p4_downsample, p5], dim=1)
        p5 = self.conv_for_downsample2(p5)

        return p3, p4, p5


class Head(nn.Module):
    """
    Detection head module.
    """

    def __init__(self, head_ch_list, num_classes):
        super(Head, self).__init__()

        self.stem1 = CBS(head_ch_list[0], head_ch_list[0], k_size=1, stride=1)
        self.stem2 = CBS(head_ch_list[1], head_ch_list[1], k_size=1, stride=1)
        self.stem3 = CBS(head_ch_list[2], head_ch_list[2], k_size=1, stride=1)

        self.cls_conv1 = nn.Sequential(
            CBS(head_ch_list[0], head_ch_list[0], k_size=3, stride=1),
            CBS(head_ch_list[0], head_ch_list[0], k_size=3, stride=1),
        )
        self.cls_conv2 = nn.Sequential(
            CBS(head_ch_list[1], head_ch_list[1], k_size=3, stride=1),
            CBS(head_ch_list[1], head_ch_list[1], k_size=3, stride=1),
        )
        self.cls_conv3 = nn.Sequential(
            CBS(head_ch_list[2], head_ch_list[2], k_size=3, stride=1),
            CBS(head_ch_list[2], head_ch_list[2], k_size=3, stride=1),
        )

        self.cls_pred1 = nn.Conv2d(head_ch_list[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.cls_pred2 = nn.Conv2d(head_ch_list[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.cls_pred3 = nn.Conv2d(head_ch_list[2], num_classes, kernel_size=1, stride=1, padding=0)

        self.reg_conv1 = nn.Sequential(
            CBS(head_ch_list[0], head_ch_list[0], k_size=3, stride=1),
            CBS(head_ch_list[0], head_ch_list[0], k_size=3, stride=1),
        )
        self.reg_conv2 = nn.Sequential(
            CBS(head_ch_list[1], head_ch_list[1], k_size=3, stride=1),
            CBS(head_ch_list[1], head_ch_list[1], k_size=3, stride=1),
        )
        self.reg_conv3 = nn.Sequential(
            CBS(head_ch_list[2], head_ch_list[2], k_size=3, stride=1),
            CBS(head_ch_list[2], head_ch_list[2], k_size=3, stride=1),
        )

        self.reg_pred1 = nn.Conv2d(head_ch_list[0], 4, kernel_size=1, stride=1, padding=0)
        self.reg_pred2 = nn.Conv2d(head_ch_list[1], 4, kernel_size=1, stride=1, padding=0)
        self.reg_pred3 = nn.Conv2d(head_ch_list[2], 4, kernel_size=1, stride=1, padding=0)

        self.obj_pred1 = nn.Conv2d(head_ch_list[0], 1, kernel_size=1, stride=1, padding=0)
        self.obj_pred2 = nn.Conv2d(head_ch_list[1], 1, kernel_size=1, stride=1, padding=0)
        self.obj_pred3 = nn.Conv2d(head_ch_list[2], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, p3, p4, p5):
        x1 = self.stem1(p3)
        cls_feat1 = self.cls_conv1(x1)
        cls_output1 = self.cls_pred1(cls_feat1)
        reg_feat1 = self.reg_conv1(x1)
        reg_output1 = self.reg_pred1(reg_feat1)
        obj_output1 = self.obj_pred1(reg_feat1)
        output1 = torch.cat([reg_output1, obj_output1, cls_output1], dim=1)

        x2 = self.stem2(p4)
        cls_feat2 = self.cls_conv2(x2)
        cls_output2 = self.cls_pred2(cls_feat2)
        reg_feat2 = self.reg_conv2(x2)
        reg_output2 = self.reg_pred2(reg_feat2)
        obj_output2 = self.obj_pred2(reg_feat2)
        output2 = torch.cat([reg_output2, obj_output2, cls_output2], dim=1)

        x3 = self.stem3(p5)
        cls_feat3 = self.cls_conv3(x3)
        cls_output3 = self.cls_pred3(cls_feat3)
        reg_feat3 = self.reg_conv3(x3)
        reg_output3 = self.reg_pred3(reg_feat3)
        obj_output3 = self.obj_pred3(reg_feat3)
        output3 = torch.cat([reg_output3, obj_output3, cls_output3], dim=1)

        return output1, output2, output3


class HumanDetectionModel(nn.Module):
    """
    Human detection model.
    """

    def __init__(self, num_classes):
        super(HumanDetectionModel, self).__init__()
        hidden_ch = 8
        block_ch = 8
        panet_ch = 8
        n_stack = 2
        ids = [-1, -2, -3, -4]
        head_ch_list = [hidden_ch * 4, hidden_ch * 8, hidden_ch * 16]

        self.backbone = Backbone(hidden_ch, block_ch, n_stack, ids)
        self.neck = Neck(hidden_ch, panet_ch, n_stack, ids)
        self.head = Head(head_ch_list, num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        output1, output2, output3 = self.head(p3, p4, p5)
        return [output1, output2, output3]


if __name__ == "__main__":
    x = torch.randn(1, 3, 480, 640)
    print("Input shape:", x.shape)

    model = HumanDetectionModel(2)
    output = model(x)

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")

    # Print the output shapes
    for i, item in enumerate(output):
        print(f"Output{i+1} shape:", item.shape)
