from .mgd import MGDLoss
from .fgd import FeatureLoss
from .fgd_mask_lvl import FeatureLoss1
from .fgd_msk_att import FeatureLoss2
from .fgd_msk_glb import FeatureLoss3
from .fgd_mask_lvl_mgd import FeatureLoss4
from .fgd_msk_channel import FeatureLoss5
from .fgd_mask_lvl_eca import FeatureLoss6
from .fgd_distill_loss import FGDDistillLoss
from .fgd_global_loss import FGDGlobalLoss
from .fgd_local_loss import FGDLocalLoss
from .mgd_distill_loss import MGDDistillLoss
from .fgd_channel_loss import FGDChannelLoss
from .fgd_spatial_loss import FGDSpatialLoss
from .fgd_original_loss import FGDOriginalLoss
from .fgd_nonlocal_loss import FGDNonlocalLoss
from .fgd_fg_bg_loss import FGDForeBackGroundLoss
from .fgd_fg_loss import FGDForeGroundLoss
from .fgd_bg_loss import FGDBackGroundLoss
from .fgd_fg_bg_split_loss import FGDForeBackGroundSplitLoss
from .fgd_channel_spatial_loss import FGDChannelSpatialLoss

from .ablation import FGD_Ori_FG_BG_Loss, FGD_Ori_BG_Loss, FGD_Ori_FG_Loss

__all__ = [
    'MGDLoss', 'FeatureLoss', 'FeatureLoss1', 'FeatureLoss2', 'FeatureLoss3', 'FeatureLoss4', 'FeatureLoss5',
    'FeatureLoss6', 'FGDDistillLoss', 'FGDGlobalLoss', 'FGDLocalLoss', 'MGDDistillLoss', 'FGDChannelLoss',
    'FGDSpatialLoss', 'FGDOriginalLoss', 'FGDNonlocalLoss', 'FGDBackGroundLoss', 'FGDForeBackGroundSplitLoss',
    'FGDForeBackGroundLoss', 'FGDForeGroundLoss',

    'FGD_Ori_FG_BG_Loss', 'FGD_Ori_BG_Loss', 'FGD_Ori_FG_Loss'
]
