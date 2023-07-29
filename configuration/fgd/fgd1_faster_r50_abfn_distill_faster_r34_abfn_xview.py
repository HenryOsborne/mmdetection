dataset_type = 'XviewDataset'
data_root = 'data/xview/'
work_dir = './work_dirs/fgd/fgd1_faster_r50_abfn_distill_faster_r50_fpn_xview'
img_scale = (800, 800)
num_classes = 1
num_proposal = 10000
student_cfg = 'configuration/tmp/faster_r50_fpn_xview.py'
teacher_cfg = 'configuration/tmp/faster_r50_abfn_xview.py'

find_unused_parameters = True
temp = 0.5
alpha_fgd = 5e-05
beta_fgd = 2.5e-05
gamma_fgd = 5e-05
lambda_fgd = 5e-07
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained='work_dirs/tmp/faster_r50_abfn_xview/epoch_30.pth',
    init_student=True,
    distill_cfg=[
        dict(
            student_module='neck.fpn_convs.3.conv',
            teacher_module='neck.fpn_convs.3.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss1',
                    name='loss_fgd_fpn_3',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd)
            ]),
        dict(
            student_module='neck.fpn_convs.2.conv',
            teacher_module='neck.fpn_convs.2.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss1',
                    name='loss_fgd_fpn_2',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd)
            ]),
        dict(
            student_module='neck.fpn_convs.1.conv',
            teacher_module='neck.fpn_convs.1.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss1',
                    name='loss_fgd_fpn_1',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd)
            ]),
        dict(
            student_module='neck.fpn_convs.0.conv',
            teacher_module='neck.fpn_convs.0.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss1',
                    name='loss_fgd_fpn_0',
                    student_channels=256,
                    teacher_channels=256,
                    temp=temp,
                    alpha_fgd=alpha_fgd,
                    beta_fgd=beta_fgd,
                    gamma_fgd=gamma_fgd,
                    lambda_fgd=lambda_fgd)
            ])
    ])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Albu', transforms = [{"type": 'RandomRotate90'}]),#数据增强
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='bbox')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 29])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
auto_resume = False
gpu_ids = [0]
