import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class FeatureLoss1(nn.Module):
    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """

    def __init__(
            self,
            student_channels,
            teacher_channels,
            name,
            temp=0.5,
            alpha_fgd=0.001,
            beta_fgd=0.0005,
            gamma_fgd=0.001,
            lambda_fgd=0.000005,
    ):
        super(FeatureLoss1, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        # ============================================================================
        self.min_size = 2  # 2 & 5 is related to anchor size (anchor-based method)
        self.max_size = 5
        self.nb_downsample = 2  # is related to the downsample times in backbone
        # ============================================================================
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))

        self.reset_parameters()

    def forward(self,
                preds_stu,
                preds_teac,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_stu(Tensor): Bs*C*H*W, student's feature map
            preds_teac(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_stu.shape[-2:] == preds_teac.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_stu = self.align(preds_stu)

        N, C, H, W = preds_stu.shape

        teac_spatial_attention, teac_channel_attention = self.get_attention(preds_teac, self.temp)
        stu_spatial_attention, stu_channel_attention = self.get_attention(preds_stu, self.temp)

        mask_fg, mask_bg = self.get_mask_lvl(teac_spatial_attention, gt_bboxes, img_metas, N, C, H, W)

        fg_loss, bg_loss = self.get_fea_loss(
            preds_stu,
            preds_teac,
            mask_fg,
            mask_bg,
            stu_channel_attention,
            teac_channel_attention,
            stu_spatial_attention,
            teac_spatial_attention)

        mask_loss = self.get_mask_loss(
            stu_channel_attention,
            teac_channel_attention,
            stu_spatial_attention,
            teac_spatial_attention)
        rela_loss = self.get_rela_loss(preds_stu, preds_teac)

        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        return loss

    def get_bbox_mask(self, area, lvl):
        min_area = 2 ** (lvl + self.min_size)
        max_area = 2 ** (lvl + self.max_size)
        if min_area < area < max_area:
            return 1
        elif lvl == 0 and area < min_area:  # scale <4  marked as 1
            return 1
        elif lvl == 3 and area > max_area:  # scale out of range  marked as 1
            return 1
        else:
            return 0
            # 一旦物体不能被第i层匹配到，则将其视作背景，用0表示
            # once the object can not be matched in i-th layer, it will be treated as background  marked as 0

    def get_mask_lvl(self, teacher_spatial_attention, gt_bboxes, img_metas, N, C, H, W):
        assert teacher_spatial_attention.size(-1) == teacher_spatial_attention.size(-2)
        shape = teacher_spatial_attention.size(-1)
        assert shape in (25, 50, 100, 200)
        lvl_dict = {'200': 0, '100': 1, '50': 2, '25': 3}
        lvl = lvl_dict[str(shape)]

        mask_foreground = torch.zeros_like(teacher_spatial_attention)
        mask_background = torch.ones_like(teacher_spatial_attention)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))
            # ============================================================================
            height, width = img_metas[i]['img_shape'][0], img_metas[i]['img_shape'][1]
            gt_mp = np.zeros((height, width))
            for j in range(len(gt_bboxes[i])):
                ann = gt_bboxes[i][j].cpu().numpy()
                x1, y1, x2, y2 = ann

                left = np.int(x1)
                top = np.int(y1)

                right = int(np.clip(np.ceil(x2), a_min=0, a_max=width - 1))
                down = int(np.clip(np.ceil(y2), a_min=0, a_max=height - 1))

                w = right - left
                h = down - top

                value = self.get_bbox_mask(np.sqrt(w * h), lvl)
                # if value == 1:
                #     gt_mp[top:down, left:right] = area[i][j].cpu().numpy()
                # gt_mp[top:down, left:right] = value
                gt_mp[top:down, left:right] = np.maximum(gt_mp[top:down, left:right], value)
            # ============================================================================

            # ----------------------------------------------------------------------------------------
            draw_label = False
            if draw_label is True:
                import os
                work_dir = 'work_dirs/tmp/'
                filename = img_metas[i]['ori_filename']
                os.makedirs(work_dir, exist_ok=True)
                sub_dir = os.path.join(work_dir, filename.split('.')[0])
                os.makedirs(sub_dir, exist_ok=True)
                name = filename.split('.')[0] + '_c' + str(lvl + 2) + '.png'

                gt_label = gt_mp.copy()
                gt_label[gt_label < 0] = 0
                gt_label = gt_label * 255.
                gt_label = np.expand_dims(gt_label, axis=-1)

                cv2.imwrite(os.path.join(sub_dir, name), gt_label)
            # ----------------------------------------------------------------------------------------

            gt_mp = cv2.resize(gt_mp, (width // (2 ** (lvl + self.nb_downsample)), height // (
                    2 ** (lvl + self.nb_downsample))))  # down sample using bilinear interpolation method
            gt_mp = gt_mp[np.newaxis, np.newaxis, :, :]
            mask_foreground[i] = torch.from_numpy(gt_mp)

            mask_background[i] = torch.where(mask_foreground[i] > 0, 0, 1)
            if torch.sum(mask_background[i]):
                mask_background[i] /= torch.sum(mask_background[i])

            return mask_foreground, mask_background

    def get_mask(self, teacher_spatial_attention, gt_bboxes, img_metas):
        mask_foreground = torch.zeros_like(teacher_spatial_attention)
        mask_background = torch.ones_like(teacher_spatial_attention)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                mask_foreground[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(mask_foreground[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            mask_background[i] = torch.where(mask_foreground[i] > 0, 0, 1)
            if torch.sum(mask_background[i]):
                mask_background[i] /= torch.sum(mask_background[i])

            return mask_foreground, mask_background

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def get_fea_loss(
            self,
            preds_stu,
            preds_teac,
            mask_fg,
            mask_bg,
            stu_channel,
            teac_channel,
            stu_spatial,
            teac_spatial):
        loss_mse = nn.MSELoss(reduction='sum')

        mask_fg = mask_fg.unsqueeze(dim=1)
        mask_bg = mask_bg.unsqueeze(dim=1)

        teac_channel = teac_channel.unsqueeze(dim=-1)
        teac_channel = teac_channel.unsqueeze(dim=-1)

        teac_spatial = teac_spatial.unsqueeze(dim=1)

        fea_t = torch.mul(preds_teac, torch.sqrt(teac_spatial))
        fea_t = torch.mul(fea_t, torch.sqrt(teac_channel))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(mask_bg))

        fea_s = torch.mul(preds_stu, torch.sqrt(teac_spatial))
        fea_s = torch.mul(fea_s, torch.sqrt(teac_channel))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        # L1 损失函数
        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

        return mask_loss

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)
