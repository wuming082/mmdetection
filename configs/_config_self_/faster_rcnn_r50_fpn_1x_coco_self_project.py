# 在文件开头添加
backend_args = None


# 添加日志级别
log_level = 'DEBUG'

# 1.首先我们是基于faster-rcnn构建的配置文件，所以我们要先继承原有的配置文件，然后做必要的修改
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py' #相对我们当前的目录，找到要继承的配置文件

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

# 1.数据加载
train_dataloader = dict(
    batch_size=3,                    # 每个批次的样本数量为4
    num_workers=2,                   # 使用2个子进程来加载数据
    persistent_workers=True,         # 在训练周期之间保持工作进程活跃，避免反复创建和销毁进程的开销
    sampler=dict(type='DefaultSampler', shuffle=True),  # 使用默认采样器，并且打乱数据顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'), # 根据图像宽高比对批次进行采样，有助于减少填充和提高训练效率
    dataset=dict(                    # 数据集配置
        type=dataset_type,           # 数据集类型（如CocoDataset、VOCDataset等）
        metainfo=metainfo,           # 数据集的元信息，通常包含类别名称等
        data_root=data_root,         # 数据集根目录路径
        ann_file='coco_framework/common/train.json',  # 标注文件路径（相对于data_root）
        data_prefix=dict(img='images/'),  # 图像文件路径前缀（相对于data_root）
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 过滤配置：过滤掉没有真实标注框和目标太小的样本
        # pipeline=train_pipeline     # 训练时的数据预处理流程（包括数据增强）
    )
)

# 2.验证集的加载
val_dataloader = dict(
    batch_size=1,                    # 验证时批次大小为1（通常验证时不需批量处理）
    num_workers=2,                   # 使用2个子进程加载数据
    persistent_workers=True,         # 保持工作进程活跃
    drop_last=False,                 # 不丢弃最后一个不完整的批次（确保所有数据都被评估）
    sampler=dict(type='DefaultSampler', shuffle=False),  # 使用默认采样器，但不打乱数据顺序
    dataset=dict(
        type=dataset_type,           # 数据集类型（应与训练集一致）
        metainfo=metainfo,           # 与训练集相同的元信息
        data_root=data_root,         # 相同的数据集根目录
        ann_file='coco_framework/common/val.json',  # 验证集标注文件路径
        data_prefix=dict(img='images/'),  # 相同的图像路径前缀
        test_mode=True,              # 设置为测试模式（不会应用训练特有的数据增强）
        # pipeline=test_pipeline      # 验证时的数据预处理流程（通常不包含随机增强）
    )
)
test_dataloader = val_dataloader  # 测试集使用与验证集相同的配置

# 2.预处理
# 训练文件的数据处理流水线
train_pipeline = [
    # 1. 从文件加载图像
    dict(type='LoadImageFromFile', backend_args=backend_args),

    # 2. 加载标注信息（边界框）
    dict(type='LoadAnnotations', with_bbox=True),

    # 3. 调整图像大小，保持宽高比
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),

    # 4. 填充图像到指定倍数
    dict(type='Pad', size_divisor=32),

    # 5. 随机翻转（当前被注释掉了）
    # dict(type='RandomFlip', prob=0.5),

    # 6. 将数据打包成模型需要的格式
    dict(type='PackDetInputs')
]
# 测试和验证的数据处理流水线
test_pipeline = [
    # 1. 从文件加载图像
    dict(type='LoadImageFromFile', backend_args=backend_args),

    # 2. 调整图像大小，保持宽高比
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),

    # 3. 加载标注信息（如果有的话）
    dict(type='LoadAnnotations', with_bbox=True),

    # 4. 将数据打包成模型需要的格式，并保留更多元信息
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 将模型的目标类型数修改为和数据集相同
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5)))

val_evaluator = dict(
    type='CocoMetric',  # 使用COCO格式的评估指标
    ann_file=data_root + 'coco_framework/common/val.json',  # 验证集标注文件路径
    metric=['bbox', 'proposal'],  # 要计算的评估指标类型
    format_only=False,  # 不只格式化结果，而是直接计算指标
    classwise=True,  # 输出每个类别的详细评估结果
    backend_args=None # 文件后端参数
)

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 使用基于训练轮次（epoch）的训练循环
    max_epochs=100,              # 最大训练轮次为100
    val_interval=1               # 每1个epoch验证一次
)

# 在配置文件末尾添加或修改优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0001), # 将 lr 调整为 0.0015
    clip_grad=None,
)

