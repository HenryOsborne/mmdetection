import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=(stride, stride),
                              padding=(padding, padding), dilation=(dilation, dilation), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, torch.sigmoid(channel_att_sum)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # --------------------------------------------------------------------------------------------
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        s, _ = torch.max(x, dim=1, keepdim=True)
        t = (x - s).exp().sum(dim=1, keepdim=True).log()
        outputs = s + t
        return outputs
        # --------------------------------------------------------------------------------------------


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return scale.squeeze(0)


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['lse'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


@DISTILL_LOSSES.register_module()
class FeatureLoss2(nn.Module):
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

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss2, self).__init__()
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
        self.teac_spatial_att = SpatialGate()
        self.teac_channel_att = ChannelGate(teacher_channels, pool_types=['lse'])
        self.stu_spatial_att = SpatialGate()
        self.stu_channel_att = ChannelGate(student_channels, pool_types=['lse'])
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

        teac_spatial_attention, teac_channel_attention = self.get_attention_teacher(preds_teac, self.temp)
        stu_spatial_attention, stu_channel_attention = self.get_attention_student(preds_stu, self.temp)

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

    def get_attention_teacher(self, preds, temp):
        x_out, channel_att = self.teac_channel_att(preds)
        spatial_att = self.teac_spatial_att(x_out)
        return spatial_att, channel_att

    def get_attention_student(self, preds, temp):
        x_out, channel_att = self.stu_channel_att(preds)
        spatial_att = self.stu_spatial_att(x_out)
        return spatial_att, channel_att

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

    def get_fea_loss(self,
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
