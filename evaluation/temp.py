from mmdet.models.utils import PatchEmbed
from mmdet.distillation.losses.fgd_msk_glb import SwinAttention
import torch

if __name__ == '__main__':
    # patch_embed = PatchEmbed(
    #     in_channels=256,
    #     embed_dims=96,
    #     conv_type='Conv2d',
    #     kernel_size=4,
    #     stride=1,
    #     norm_cfg=dict(type='LN'),
    #     init_cfg=None)
    # x = torch.randn(1, 256, 200, 200)
    # y, y_size = patch_embed(x)
    # pass

    att = SwinAttention(256, 96, name='loss_fgd_fpn_3')
    x = torch.randn(1, 256, 25, 25)
    y = att(x)
    pass
