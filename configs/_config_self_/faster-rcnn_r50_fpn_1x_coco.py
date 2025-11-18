_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# ✅ 你的完整配置
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5)  # 你的5个类别
    )
)

# 数据加载器配置（在训练脚本中动态修改路径）
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDataset',
        data_root='/home/jovyan/workspace/YOLO_DATASET/',  # 运行时修改
        ann_file='train.json',   # 运行时修改
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDataset',
        data_root='/home/jovyan/workspace/YOLO_DATASET/',  # 运行时修改
        ann_file='val.json',     # 运行时修改
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False
)

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,  # 你的要求
    val_interval=1
)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[80, 90], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

