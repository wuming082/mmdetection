model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        out_indices=(1, 2, 4, 6),
        frozen_stages=-1,
        init_cfg=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    # 关键：添加 train_cfg 和 test_cfg
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# 数据集设置 (以 COCO 为例)
dataset_type = 'CocoDataset'
data_root = 'data/coco/' # 你的数据集根目录

# 训练管道
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True), # 输入尺寸
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 验证管道
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True), # 输入尺寸
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 训练数据加载器
train_dataloader = dict(
    batch_size=16, # 根据你的 GPU 内存调整
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None))

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

# 测试数据加载器
test_dataloader = val_dataloader

# 验证评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

# 测试评估器
test_evaluator = val_evaluator

# 训练循环配置 - 注意这里保留，但 RetinaHead 的 train_cfg 优先级更高
train_cfg = dict(
    type='EpochBasedTrainLoop', # 基于 epoch 的训练循环
    max_epochs=100, # 训练 100 轮次
    val_interval=1) # 每个 epoch 验证一次

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001), # 学习率设为 0.015
    clip_grad=dict(max_norm=35, norm_type=2))

# 学习率调度器配置
param_scheduler = [
    dict(
        type='LinearLR', # 线性预热
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500), # 预热 500 个 iteration
    dict(
        type='MultiStepLR', # 多步衰减
        by_epoch=True, # 按 epoch 调整
        begin=0,
        end=100, # 总共 100 个 epoch
        milestones=[80], # 在第 80 个 epoch 时衰减
        gamma=0.1) # 衰减因子
]

# 默认运行时设置
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1), # 每个 epoch 保存一次
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None # 从头开始训练
resume = False # 不恢复训练