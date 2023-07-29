# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
import warnings
import mmcv
import torch
from mmcv.runner import load_checkpoint
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .. import build_detector


@DETECTORS.register_module()
class KnowledgeDistillationTwoStageDetector(TwoStageDetector):
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
        # =====================================================================
        x = self.extract_feat(img)

        losses = dict()

        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            proposal_list_teacher = self.teacher_model.rpn_head.simple_test_rpn(teacher_x, img_metas)
            out_rpn = self.teacher_model.rpn_head.simple_test_rpn_teacher(teacher_x, gt_bboxes, gt_labels, img_metas)
            out_teacher = self.teacher_model.roi_head.simple_test_bboxes_teacher(
                x, img_metas, proposal_list_teacher, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal',
                self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                out_rpn,
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
            x, img_metas, proposal_list, out_teacher,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks,
            **kwargs)
        losses.update(roi_losses)

        return losses
        # =====================================================================

        # x = self.extract_feat(img)
        #
        # losses = dict()
        #
        # with torch.no_grad():
        #     teacher_x = self.teacher_model.extract_feat(img)
        #     proposal_list_teacher = self.teacher_model.rpn_head.simple_test_rpn(teacher_x, img_metas)
        #     out_rpn = self.teacher_model.rpn_head.simple_test_rpn_teacher(teacher_x, gt_bboxes, gt_labels, img_metas)
        #     out_teacher = self.teacher_model.roi_head.simple_test_bboxes_teacher(
        #         x, img_metas, proposal_list_teacher, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks)
        #
        # # RPN forward and loss
        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get(
        #         'rpn_proposal',
        #         self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         gt_bboxes,
        #         gt_labels=None,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg,
        #         **kwargs)
        #     losses.update(rpn_losses)
        # else:
        #     proposal_list = proposals
        #
        # roi_losses = self.roi_head.forward_train(
        #     x, img_metas, proposal_list, out_teacher,
        #     gt_bboxes, gt_labels,
        #     gt_bboxes_ignore, gt_masks,
        #     **kwargs)
        # losses.update(roi_losses)
        #
        # return losses

        # =====================================================================

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
