# 在文件开头添加
backend_args = None

# 添加日志级别
log_level = 'DEBUG'

# 指向你的 MobileNetV2 基础配置文件
_base_ = '../mobilenet/sdd-mobilenetv2-fpn-coco-x1.py' # <--- 修改这里

# 2.声明我们使用数据类型
dataset_type = 'CocoDataset' 
# 3.配置数据根目录
data_root = '/home/jovyan/workspace/YOLO_DATASET/' # 数据集目录

# 4.修改类别，检测框颜色等信息
metainfo = {
    'classes': ('red', 'green', 'crosswalk', 'blind', 'pothole'),
    'palette': [
        (255, 0, 0),      # red - 红色
        (0, 255, 0),      # green - 绿色  
        (255, 255, 0),    # crosswalk - 黄色
        (0, 0, 255),      # blind - 蓝色
        (128, 0, 128)     # pothole - 紫色
    ]
}

# (如果需要自定义) 定义训练时的数据预处理流程
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True), # 示例尺寸
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# (如果需要自定义) 定义验证/测试时的数据预处理流程
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True), # 示例尺寸
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 关键修改：覆盖类别数，并显式保留 train_cfg 和 test_cfg
model = dict(
    bbox_head=dict(
        num_classes=5,
    )
)

# 1.数据加载
train_dataloader = dict(
    batch_size=3,                    # 每个批次的样本数量
    num_workers=2,                   # 使用2个子进程来加载数据
    persistent_workers=True,         # 在训练周期之间保持工作进程活跃
    sampler=dict(type='DefaultSampler', shuffle=True),  # 使用默认采样器，并且打乱数据顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'), # 根据图像宽高比对批次进行采样
    dataset=dict(                    # 数据集配置
        type=dataset_type,           # 数据集类型
        metainfo=metainfo,           # 数据集的元信息，包含类别名称和颜色
        data_root=data_root,         # 数据集根目录路径
        ann_file='coco_framework/common/train.json',  # 标注文件路径（相对于data_root）
        data_prefix=dict(img='images/'),  # 图像文件路径前缀（相对于data_root）
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 过滤配置
        pipeline=train_pipeline     # 训练时的数据预处理流程
    )
)

# 2.验证集的加载
val_dataloader = dict(
    batch_size=1,                    # 验证时批次大小
    num_workers=2,                   # 使用2个子进程加载数据
    persistent_workers=True,         # 保持工作进程活跃
    drop_last=False,                 # 不丢弃最后一个不完整的批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 使用默认采样器，但不打乱数据顺序
    dataset=dict(
        type=dataset_type,           # 数据集类型（应与训练集一致）
        metainfo=metainfo,           # 与训练集相同的元信息
        data_root=data_root,         # 相同的数据集根目录
        ann_file='coco_framework/common/val.json',  # 验证集标注文件路径
        data_prefix=dict(img='images/'),  # 相同的图像路径前缀
        test_mode=True,              # 设置为测试模式
        pipeline=test_pipeline      # 验证时的数据预处理流程
    )
)

test_dataloader = val_dataloader # 修复拼写错误

# 验证评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco_framework/common/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

# 测试评估器
test_evaluator = val_evaluator

# 训练循环配置
train_cfg = dict(
    type='EpochBasedTrainLoop', # 基于 epoch 的训练循环
    max_epochs=500, # 训练 500 轮次
    val_interval=1) # 每个 epoch 验证一次

# 在继承文件中修改 default_hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,  # 每10个epoch保存一次
        by_epoch=True,
        save_best='coco/bbox_mAP_50',  # 或者 'coco/bbox_mAP'
        rule='greater',
        max_keep_ckpts=2,  # 最多保留2个检查点
        save_last=True,     # 保存最后一个epoch
        save_optimizer=False,  # 不保存优化器（显著节省空间）
        save_param_scheduler=False  # 不保存调度器
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
# 注意：其他配置项（如 default_scope, env_cfg, vis_backends 等）
# 如果在 _base_ 文件 (retinanet-mobilenetv2-fpn_100e_custom.py) 中已定义，这里不需要重复定义，除非需要修改。