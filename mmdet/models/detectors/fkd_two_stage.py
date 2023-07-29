# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import warnings
import mmcv
import torch
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .. import build_detector
import torch.nn as nn
from mmdet.distillation.builder import build_distill_loss
from collections import OrderedDict
import matplotlib.pyplot as plt


class NonLocalBlockND(nn.Module):
    def __init__(
            self,
            in_channels,
            inter_channels=None,
            dimension=2,
            sub_sample=True,
            bn_layer=True,
            downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)  # 2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # 2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)  # 2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # 2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)  # 2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # 2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)  # 2 , 300x300 , 150x150
        N = f.size(-1)  # 150 x 150
        f_div_C = f / N  # 2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  # 2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous()  # 2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


def plot_attention_mask(mask):
    mask = torch.squeeze(mask, dim=0)
    mask = mask.cpu().detach().numpy()
    plt.imshow(mask)
    plt.plot(mask)
    plt.savefig('1.png')
    print('saved')
    input()


@DETECTORS.register_module()
class FKD_TwoStageDetector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(
            self,
            backbone,
            neck=None,
            rpn_head=None,
            roi_head=None,
            teacher_config=None,
            teacher_ckpt=None,
            eval_teacher=True,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        # ======================================================================
        self.adaptation_type = '1x1conv'

        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])

        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )

        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])
        # ======================================================================
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

    def get_teacher_info(
            self,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            proposals=None,
            t_feats=None,
            **kwargs):
        teacher_info = {}
        x = self.extract_feat(img)
        teacher_info.update({'feat': x})
        # RPN forward and loss
        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        #     rpn_losses, proposal_list, rpn_outs = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         gt_bboxes,
        #         gt_labels=None,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg)
        #     teacher_info.update({'proposal_list': proposal_list})
        #     #   teacher_info.update({'rpn_out': rpn_outs})
        # else:
        #     proposal_list = proposals
        #
        # roi_losses, roi_out = self.roi_head.forward_train(
        #     x, img_metas, proposal_list,
        #     gt_bboxes, gt_labels,
        #     gt_bboxes_ignore, gt_masks, get_out=True,
        #     **kwargs)
        # teacher_info.update(
        #     cls_score=roi_out['cls_score'],
        #     pos_index=roi_out['pos_index'],
        #     bbox_pred=roi_out['bbox_pred'],
        #     labels=roi_out['labels'],
        #     bbox_feats=roi_out['bbox_feats'],
        #     x_cls=roi_out['x_cls'],
        #     x_reg=roi_out['x_reg']
        # )

        return teacher_info

    def with_student_proposal(
            self,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            proposals=None,
            s_info=None,
            t_info=None,
            **kwargs):

        with torch.no_grad():
            _, t_roi_out = self.roi_head.forward_train(
                t_info['feat'], img_metas, s_info['proposal_list'],
                gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, get_out=True,
                **kwargs)

        t_cls, s_cls, pos_index, labels = t_roi_out['cls_score'], s_info['cls_score'], t_roi_out[
            'pos_index'], t_roi_out['labels']
        t_cls_pos, s_cls_pos, labels_pos = t_cls[pos_index.type(torch.bool)], s_cls[pos_index.type(torch.bool)], labels[
            pos_index.type(torch.bool)]
        teacher_prediction = torch.max(t_cls_pos, dim=1)[1]
        correct_index = (teacher_prediction == labels_pos).detach()
        t_cls_pos_correct, s_cls_pos_correct = t_cls_pos[correct_index], s_cls_pos[correct_index]
        kd_pos_cls_loss = CrossEntropy(s_cls_pos_correct, t_cls_pos_correct) * 0.005
        kd_loss = dict(kd_pos_cls_loss=kd_pos_cls_loss)
        return kd_loss

    def forward_train(
            self,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=None,
            gt_masks=None,
            proposals=None,
            **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        losses = dict()

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_x = self.teacher_model.extract_feat(img)

        t = 0.5
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0

        #   for channel attention
        c_t = 0.5
        c_s_ratio = 1.0

        if teacher_x is not None:
            t_feats = teacher_x
            for _i in range(len(t_feats)):
                t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
                size = t_attention_mask.size()
                t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
                t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
                t_attention_mask = t_attention_mask.view(size)

                s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size = s_attention_mask.size()
                s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
                s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
                s_attention_mask = s_attention_mask.view(size)

                c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_t_attention_mask.size()
                c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
                c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_s_attention_mask.size()
                c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
                c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
                sum_attention_mask = sum_attention_mask.detach()

                c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask = c_sum_attention_mask.detach()

                kd_feat_loss += dist2(
                    t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask,
                    channel_attention_mask=None) * 7e-5

                kd_channel_loss += torch.dist(
                    torch.mean(t_feats[_i], [2, 3]),
                    self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3
                t_spatial_pool = torch.mean(
                    t_feats[_i], [1]).view(
                    t_feats[_i].size(0), 1, t_feats[_i].size(2),
                    t_feats[_i].size(3))
                s_spatial_pool = torch.mean(
                    x[_i], [1]).view(
                    x[_i].size(0), 1, x[_i].size(2),
                    x[_i].size(3))

                kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        kd_nonlocal_loss = 0
        if teacher_x is not None:
            t_feats = teacher_x
            for _i in range(len(t_feats)):
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                #   print(s_relation.size())
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
        losses.update(kd_nonlocal_loss=kd_nonlocal_loss * 7e-5)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal',
                self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks,
            **kwargs)
        losses.update(roi_losses)

        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
