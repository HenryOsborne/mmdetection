# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import warnings
import mmcv
import torch
from mmcv.runner import load_checkpoint
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .. import build_detector
import torch.nn as nn


class FeatureAdapLayer(nn.Module):
    def __init__(self, channels):
        super(FeatureAdapLayer, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.layer(x)


@DETECTORS.register_module()
class FGFI_TwoStageDetector(TwoStageDetector):
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
        # 控制Teacher模型预测的个数
        # teacher_config['model']['test_cfg']['rpn']['max_per_img'] = 512
        out_level = neck['num_outs']
        out_channels = neck['out_channels']
        self.imitation_loss_weigth = 0.01
        self.stu_feature_adap = []
        for i in range(out_level):
            self.stu_feature_adap.append(
                FeatureAdapLayer(out_channels)
            )

        for layer in self.stu_feature_adap:
            layer.cuda()
        # ======================================================================
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

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

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal',
                self.test_cfg.rpn)
            rpn_losses, proposal_list, mask_batch = self.rpn_head.forward_train(
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

        # ==============================================================
        # FGFI loss function
        # imitation_loss_weigth = 0.01
        # mask_list = []
        # for mask in mask_batch:
        #     mask = (mask > 0).float().unsqueeze(0)
        #     mask_list.append(mask)
        # mask_batch = torch.stack(mask_list, dim=0)
        # norms = mask_batch.sum() * 2
        #
        # sup_loss = (torch.pow(sup_feature - stu_feature_adap, 2) * mask_batch).sum() / norms
        # sup_loss = sup_loss * imitation_loss_weigth
        # ==============================================================

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks,
            **kwargs)
        losses.update(roi_losses)

        # ==============================================================
        # FGFI loss function
        fgfi_loss = []
        for i, (stu_feature, sup_feature, mask) in enumerate(zip(x, teacher_x, mask_batch)):
            # stu_feature = stu_feature.cpu()
            # sup_feature = sup_feature.cpu()
            # mask = mask.cpu()
            stu_feature = self.stu_feature_adap[i](stu_feature)
            mask = (mask > 0).float().unsqueeze(0)
            norms = mask.sum() * 2

            sup_loss = (torch.pow(sup_feature - stu_feature, 2) * mask).sum() / norms
            sup_loss = sup_loss * self.imitation_loss_weigth
            fgfi_loss.append(sup_loss)
        fgfi_loss.pop()
        fgfi_loss.pop()
        # ==============================================================
        losses.update({'fgfi_loss': fgfi_loss})

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
