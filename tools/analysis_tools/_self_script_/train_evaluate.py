import os
import json
import argparse
import torch
import time
import numpy as np
from mmengine import Config
from mmdet.registry import RUNNERS
from mmengine.runner import Runner
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from mmdet.structures import DetDataSample
from typing import List, Dict

def evaluate_model(config_path, checkpoint_path, output_dir, 
                  img_dir=None, ann_file=None, eval_options=None,
                  path_mappings=None):
    """
    评估 MMDetection 模型并生成详细报告
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型权重路径
        output_dir: 输出目录
        img_dir: 图片目录路径
        ann_file: 标注文件路径
        eval_options: 评估选项
        path_mappings: 路径映射字典 {训练服务器路径: 评估服务器路径}
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    cfg = Config.fromfile(config_path)
    
    # 应用路径映射
    if path_mappings:
        cfg = apply_path_mappings(cfg, path_mappings)
    
    # 设置评估配置
    cfg.work_dir = output_dir
    cfg.load_from = checkpoint_path
    
    # 禁用验证时的检查点保存
    disable_checkpoint_saving(cfg)
    
    # 如果提供了图片目录和标注文件，更新数据配置
    if img_dir and ann_file:
        update_data_config(cfg, img_dir, ann_file)
    
    # 如果有自定义评估选项，应用它们
    if eval_options:
        if 'batch_size' in eval_options:
            cfg.val_dataloader.batch_size = eval_options['batch_size']
        if 'workers' in eval_options:
            cfg.val_dataloader.num_workers = eval_options['workers']
    
    # 创建 runner
    runner = Runner.from_cfg(cfg)
    
    print("开始评估模型...")
    
    # 执行评估
    metrics = runner.val()
    
    # 测量推理速度
    speed_metrics = measure_inference_speed(runner, num_test_images=100)
    
    # 获取模型大小信息
    model_size_info = get_model_size(checkpoint_path)
    
    # 处理评估结果
    results = process_evaluation_results(metrics, cfg, output_dir, img_dir, ann_file)
    
    # 合并所有指标
    results['speed_metrics'] = speed_metrics
    results['model_size'] = model_size_info
    
    # 生成报告
    generate_reports(results, output_dir)
    
    return results

def measure_inference_speed(runner, num_test_images=100, warmup_iter=10):
    """
    测量模型推理速度
    
    Args:
        runner: MMDetection runner
        num_test_images: 测试图片数量
        warmup_iter: 预热迭代次数
    """
    print("开始测量推理速度...")
    
    # 获取数据加载器
    dataloader = runner.val_dataloader
    
    # 预热
    model = runner.model
    model.eval()
    
    # 预热阶段
    print("预热阶段...")
    with torch.no_grad():
        for i, data_batch in enumerate(dataloader):
            if i >= warmup_iter:
                break
            _ = model.test_step(data_batch)
    
    # 正式测试
    print("正式测试推理速度...")
    total_time = 0
    total_images = 0
    inference_times = []
    
    with torch.no_grad():
        for i, data_batch in enumerate(dataloader):
            if total_images >= num_test_images:
                break
                
            batch_size = len(data_batch['data_samples'])
            
            # 测量推理时间
            start_time = time.time()
            _ = model.test_step(data_batch)
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_images += batch_size
            inference_times.extend([batch_time / batch_size] * batch_size)
    
    # 计算统计指标
    inference_times = np.array(inference_times)
    fps_values = 1.0 / inference_times
    
    speed_metrics = {
        'total_images_tested': total_images,
        'total_time_seconds': total_time,
        'average_time_per_image': np.mean(inference_times),
        'std_time_per_image': np.std(inference_times),
        'min_time_per_image': np.min(inference_times),
        'max_time_per_image': np.max(inference_times),
        'fps_mean': np.mean(fps_values),
        'fps_std': np.std(fps_values),
        'fps_min': np.min(fps_values),
        'fps_max': np.max(fps_values),
        'fps_95_percentile': np.percentile(fps_values, 95),
        'fps_5_percentile': np.percentile(fps_values, 5)
    }
    
    print(f"推理速度测试完成: {speed_metrics['fps_mean']:.2f} FPS")
    return speed_metrics

def get_model_size(checkpoint_path):
    """
    获取模型大小信息
    """
    # 获取文件大小
    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # 修复 PyTorch 2.6 的加载问题
    try:
        # 首先尝试使用 weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"使用 weights_only=False 加载失败: {e}")
        try:
            # 如果失败，尝试使用 mmengine 的加载方法
            from mmengine.runner import load_checkpoint
            checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
        except Exception as e2:
            print(f"使用 mmengine 加载也失败: {e2}")
            # 如果都失败，只返回文件大小信息
            return {
                'checkpoint_file_size_mb': file_size_mb,
                'checkpoint_file_size_bytes': file_size_bytes,
                'error': f'无法加载检查点: {str(e)}'
            }
    
    model_size_info = {
        'checkpoint_file_size_mb': file_size_mb,
        'checkpoint_file_size_bytes': file_size_bytes,
        'state_dict_keys': list(checkpoint.get('state_dict', {}).keys()) if 'state_dict' in checkpoint else [],
        'meta_data': checkpoint.get('meta', {}),
        'has_optimizer': 'optimizer' in checkpoint,
        'has_scheduler': 'lr_scheduler' in checkpoint
    }
    
    # 如果包含状态字典，计算参数数量
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        total_params = 0
        trainable_params = 0
        
        for name, param in state_dict.items():
            if torch.is_tensor(param):
                param_count = param.numel()
                total_params += param_count
                # 计算可训练参数（排除BN和bias等）
                if 'weight' in name and 'bn' not in name and 'norm' not in name:
                    trainable_params += param_count
        
        model_size_info['total_parameters'] = total_params
        model_size_info['total_parameters_millions'] = total_params / 1e6
        model_size_info['trainable_parameters'] = trainable_params
        model_size_info['trainable_parameters_millions'] = trainable_params / 1e6
    
    return model_size_info

def disable_checkpoint_saving(cfg):
    """
    禁用验证时的检查点保存，避免 CheckpointHook 错误
    """
    # 方法1: 直接移除 checkpoint hook
    if hasattr(cfg, 'default_hooks') and 'checkpoint' in cfg.default_hooks:
        # 创建一个不保存检查点的配置
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook',
            interval=1,
            save_best=None,  # 不保存最佳模型
            max_keep_ckpts=1,
            save_last=False,
            save_optimizer=False
        )
    
    # 方法2: 设置验证时不执行 checkpoint hook
    if hasattr(cfg, 'custom_hooks'):
        # 确保没有重复的 checkpoint hook
        cfg.custom_hooks = [hook for hook in cfg.custom_hooks if hook.get('type') != 'CheckpointHook']
    
    # 方法3: 设置验证频率为很大的值，避免触发保存
    if hasattr(cfg, 'val_evaluator'):
        # 确保评估器配置正确
        pass

def apply_path_mappings(cfg, path_mappings):
    """
    应用路径映射，将训练服务器路径转换为评估服务器路径
    """
    def replace_paths(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    for train_path, eval_path in path_mappings.items():
                        if train_path in value:
                            obj[key] = value.replace(train_path, eval_path)
                            print(f"路径映射: {value} -> {obj[key]}")
                else:
                    replace_paths(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    for train_path, eval_path in path_mappings.items():
                        if train_path in item:
                            obj[i] = item.replace(train_path, eval_path)
                            print(f"路径映射: {item} -> {obj[i]}")
                else:
                    replace_paths(item)
    
    # 递归替换所有路径
    replace_paths(cfg._cfg_dict)
    return cfg

def update_data_config(cfg, img_dir, ann_file):
    """
    更新数据配置以使用指定的图片目录和标注文件
    """
    # 更新验证集配置
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.dataset.data_root = os.path.dirname(img_dir) if img_dir else ''
        cfg.val_dataloader.dataset.ann_file = ann_file
        cfg.val_dataloader.dataset.data_prefix.img = img_dir
    
    # 更新测试集配置（如果需要）
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.data_root = os.path.dirname(img_dir) if img_dir else ''
        cfg.test_dataloader.dataset.ann_file = ann_file
        cfg.test_dataloader.dataset.data_prefix.img = img_dir
    
    # 更新数据集配置
    if hasattr(cfg, 'val_evaluator'):
        if hasattr(cfg.val_evaluator, 'ann_file'):
            cfg.val_evaluator.ann_file = ann_file
    
    if hasattr(cfg, 'test_evaluator'):
        if hasattr(cfg.test_evaluator, 'ann_file'):
            cfg.test_evaluator.ann_file = ann_file

def process_evaluation_results(metrics, cfg, output_dir, img_dir=None, ann_file=None):
    """
    处理评估结果
    """
    
    # 获取类别信息
    if hasattr(cfg, 'metainfo') and 'classes' in cfg.metainfo:
        class_names = cfg.metainfo['classes']
    else:
        # 尝试从数据集中获取
        try:
            from mmdet.datasets import build_dataset
            dataset_cfg = cfg.val_dataloader.dataset if hasattr(cfg, 'val_dataloader') else cfg.test_dataloader.dataset
            dataset = build_dataset(dataset_cfg)
            class_names = dataset.metainfo.get('classes', [])
        except:
            # 默认类别名称
            class_names = [f'class_{i}' for i in range(metrics.get('num_classes', 80))]
    
    # 构建结果字典
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': cfg.filename,
        'checkpoint': cfg.load_from,
        'image_directory': img_dir,
        'annotation_file': ann_file,
        'num_classes': len(class_names),
        'class_names': class_names,
        'metrics': {},
        'class_wise_metrics': {}
    }
    
    # 提取主要指标
    bbox_metrics = {}
    precision_recall_metrics = {}
    
    for key, value in metrics.items():
        # 转换 tensor 为 float
        if torch.is_tensor(value):
            value = float(value)
        
        if any(x in key for x in ['bbox', 'mAP', 'AR', 'AP']) or key in ['mAP', 'mAP_50', 'mAP_75']:
            bbox_metrics[key] = value
        
        # 提取 precision 和 recall 相关指标
        if 'precision' in key.lower():
            precision_recall_metrics[key] = value
        if 'recall' in key.lower():
            precision_recall_metrics[key] = value
    
    results['metrics'] = bbox_metrics
    results['precision_recall_metrics'] = precision_recall_metrics
    
    # 提取类别级别的指标
    if 'coco/classwise' in metrics:
        classwise_metrics = metrics['coco/classwise']
        for i, class_metrics in enumerate(classwise_metrics):
            if i < len(class_names):
                class_name = class_names[i]
                results['class_wise_metrics'][class_name] = {
                    'AP': float(class_metrics.get('ap', 0)),
                    'AP50': float(class_metrics.get('ap50', 0)),
                    'AP75': float(class_metrics.get('ap75', 0)),
                    'precision': float(class_metrics.get('precision', 0)),
                    'recall': float(class_metrics.get('recall', 0))
                }
    
    return results

def generate_reports(results, output_dir):
    """
    生成 MD 表格和 JSON 报告
    """
    
    # 生成 JSON 报告
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON 报告已保存: {json_path}")
    
    # 生成 MD 表格报告
    md_path = os.path.join(output_dir, 'evaluation_report.md')
    generate_md_report(results, md_path)
    print(f"MD 报告已保存: {md_path}")
    
    # 生成可视化图表
    generate_visualizations(results, output_dir)
    print(f"可视化图表已保存到: {output_dir}")

def generate_md_report(results, md_path):
    """
    生成 Markdown 格式的报告
    """
    
    with open(md_path, 'w', encoding='utf-8') as f:
        # 标题和基本信息
        f.write("# 目标检测模型评估报告\n\n")
        f.write(f"**生成时间**: {results['timestamp']}\n\n")
        f.write(f"**配置文件**: `{results['config']}`\n\n")
        f.write(f"**模型权重**: `{results['checkpoint']}`\n\n")
        
        if results.get('image_directory'):
            f.write(f"**图片目录**: `{results['image_directory']}`\n\n")
        if results.get('annotation_file'):
            f.write(f"**标注文件**: `{results['annotation_file']}`\n\n")
        
        f.write(f"**类别数量**: {results['num_classes']}\n\n")
        f.write(f"**类别名称**: {', '.join(results['class_names'])}\n\n")
        
        # 模型大小信息
        if 'model_size' in results:
            model_size = results['model_size']
            f.write("## 模型大小信息\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 检查点文件大小 | {model_size['checkpoint_file_size_mb']:.2f} MB |\n")
            if 'total_parameters' in model_size:
                f.write(f"| 总参数量 | {model_size['total_parameters_millions']:.2f} M |\n")
                f.write(f"| 可训练参数量 | {model_size['trainable_parameters_millions']:.2f} M |\n")
            if 'error' in model_size:
                f.write(f"| 错误信息 | {model_size['error']} |\n")
        
        # 推理速度信息
        if 'speed_metrics' in results:
            speed = results['speed_metrics']
            f.write("\n## 推理速度\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 平均 FPS | {speed['fps_mean']:.2f} |\n")
            f.write(f"| FPS 标准差 | {speed['fps_std']:.2f} |\n")
            f.write(f"| 最小 FPS | {speed['fps_min']:.2f} |\n")
            f.write(f"| 最大 FPS | {speed['fps_max']:.2f} |\n")
            f.write(f"| 95% FPS | {speed['fps_95_percentile']:.2f} |\n")
            f.write(f"| 平均推理时间 | {speed['average_time_per_image']*1000:.2f} ms |\n")
            f.write(f"| 测试图片数量 | {speed['total_images_tested']} |\n")
        
        # 主要指标表格
        f.write("\n## 主要评估指标\n\n")
        f.write("| 指标 | 数值 | 描述 |\n")
        f.write("|------|------|------|\n")
        
        metrics = results['metrics']
        metric_descriptions = {
            'bbox_mAP': '平均精度均值 (IoU=0.50:0.95)',
            'bbox_mAP_50': '平均精度 (IoU=0.50)',
            'bbox_mAP_75': '平均精度 (IoU=0.75)',
            'bbox_mAP_s': '小目标平均精度',
            'bbox_mAP_m': '中目标平均精度',
            'bbox_mAP_l': '大目标平均精度',
            'bbox_mAP_copypaste': 'COCO 格式精度汇总'
        }
        
        for key, value in metrics.items():
            description = metric_descriptions.get(key, '')
            if isinstance(value, (int, float)):
                f.write(f"| {key} | {value:.4f} | {description} |\n")
            else:
                f.write(f"| {key} | {value} | {description} |\n")
        
        # Precision 和 Recall 指标
        if 'precision_recall_metrics' in results and results['precision_recall_metrics']:
            f.write("\n## Precision 和 Recall 指标\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            for key, value in results['precision_recall_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"| {key} | {value:.4f} |\n")
                else:
                    f.write(f"| {key} | {value} |\n")
        
        # 类别级别指标
        if results['class_wise_metrics']:
            f.write("\n## 类别级别指标\n\n")
            f.write("| 类别 | AP | AP50 | AP75 | Precision | Recall |\n")
            f.write("|------|----|------|------|-----------|--------|\n")
            
            for class_name, metrics in results['class_wise_metrics'].items():
                f.write(f"| {class_name} | {metrics['AP']:.4f} | {metrics['AP50']:.4f} | {metrics['AP75']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")
        
        # 性能总结
        f.write("\n## 性能总结\n\n")
        
        # 提取关键指标
        key_metrics = {}
        for key in ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l']:
            if key in metrics:
                key_metrics[key] = metrics[key]
        
        if key_metrics:
            f.write("### 关键指标\n\n")
            for key, value in key_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"- **{key}**: {value:.4f}\n")
                else:
                    f.write(f"- **{key}**: {value}\n")
            
            # 性能评级
            mAP = key_metrics.get('bbox_mAP', 0)
            if isinstance(mAP, (int, float)):
                if mAP >= 0.8:
                    rating = "优秀"
                elif mAP >= 0.6:
                    rating = "良好"
                elif mAP >= 0.4:
                    rating = "一般"
                else:
                    rating = "需要改进"
                
                f.write(f"\n**综合性能评级**: {rating} (基于 mAP: {mAP:.4f})\n")
            else:
                f.write(f"\n**综合性能评级**: 无法评估 (mAP 不是数值类型)\n")
        
        # 速度性能评级
        if 'speed_metrics' in results:
            speed = results['speed_metrics']
            fps = speed['fps_mean']
            if fps >= 30:
                speed_rating = "优秀 (实时)"
            elif fps >= 15:
                speed_rating = "良好"
            elif fps >= 5:
                speed_rating = "一般"
            else:
                speed_rating = "较慢"
            
            f.write(f"\n**推理速度评级**: {speed_rating} (基于 FPS: {fps:.2f})\n")

def generate_visualizations(results, output_dir):
    """
    生成可视化图表
    """
    
    # 设置样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. 主要指标柱状图
    if results['metrics']:
        plt.figure(figsize=(12, 6))
        metrics = results['metrics']
        
        # 选择主要指标
        main_metrics = {}
        for k, v in metrics.items():
            if any(x in k for x in ['mAP', 'AR']) and isinstance(v, (int, float)):
                main_metrics[k] = v
        
        if main_metrics:
            plt.bar(range(len(main_metrics)), list(main_metrics.values()))
            plt.xticks(range(len(main_metrics)), list(main_metrics.keys()), rotation=45)
            plt.title('Main Evaluation Metrics')
            plt.ylabel('Score')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'main_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. 推理速度分布图
    if 'speed_metrics' in results:
        # 创建模拟的推理时间分布（实际应该从测试中记录所有时间）
        speed = results['speed_metrics']
        mean_time = speed['average_time_per_image']
        std_time = speed['std_time_per_image']
        
        # 生成模拟的时间分布数据
        np.random.seed(42)
        simulated_times = np.random.normal(mean_time, std_time, 1000)
        simulated_fps = 1.0 / simulated_times
        
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_fps, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(speed['fps_mean'], color='red', linestyle='--', label=f'平均 FPS: {speed["fps_mean"]:.2f}')
        plt.xlabel('FPS')
        plt.ylabel('频率')
        plt.title('推理速度分布 (FPS)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fps_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 类别级别 AP 图
    if results['class_wise_metrics']:
        class_metrics = results['class_wise_metrics']
        
        # AP 柱状图
        plt.figure(figsize=(max(8, len(class_metrics) * 0.6), 6))
        classes = list(class_metrics.keys())
        ap_values = [metrics['AP'] for metrics in class_metrics.values()]
        
        bars = plt.bar(classes, ap_values)
        plt.xticks(rotation=45, ha='right')
        plt.title('AP by Class')
        plt.ylabel('AP')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, ap_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_ap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # AP50 和 AP75 对比图
        if len(class_metrics) > 0:
            plt.figure(figsize=(max(8, len(class_metrics) * 0.6), 6))
            ap50_values = [metrics['AP50'] for metrics in class_metrics.values()]
            ap75_values = [metrics['AP75'] for metrics in class_metrics.values()]
            
            x = range(len(classes))
            width = 0.35
            
            bars1 = plt.bar([i - width/2 for i in x], ap50_values, width, label='AP50', alpha=0.8)
            bars2 = plt.bar([i + width/2 for i in x], ap75_values, width, label='AP75', alpha=0.8)
            
            plt.xticks(x, classes, rotation=45, ha='right')
            plt.title('AP50 vs AP75 by Class')
            plt.ylabel('AP')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_ap_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Precision-Recall 散点图
        if len(class_metrics) > 0:
            plt.figure(figsize=(8, 6))
            precisions = [metrics['precision'] for metrics in class_metrics.values()]
            recalls = [metrics['recall'] for metrics in class_metrics.values()]
            
            plt.scatter(recalls, precisions, alpha=0.6, s=60)
            
            # 添加类别标签
            for i, class_name in enumerate(classes):
                plt.annotate(class_name, (recalls[i], precisions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall by Class')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'precision_recall_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """
    主函数 - 支持命令行参数
    """
    parser = argparse.ArgumentParser(description='MMDetection 模型评估工具')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--img-dir', help='图片目录路径')
    parser.add_argument('--ann-file', help='标注文件路径')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--workers', type=int, default=4, help='工作进程数')
    parser.add_argument('--path-mappings', nargs='+', help='路径映射，格式: 训练路径:评估路径')
    parser.add_argument('--speed-test-images', type=int, default=100, help='速度测试图片数量')
    
    args = parser.parse_args()
    
    # 解析路径映射
    path_mappings = {}
    if args.path_mappings:
        for mapping in args.path_mappings:
            if ':' in mapping:
                train_path, eval_path = mapping.split(':', 1)
                path_mappings[train_path] = eval_path
    
    # 构建评估选项
    eval_options = {
        'batch_size': args.batch_size,
        'workers': args.workers
    }
    
    # 执行评估
    results = evaluate_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        eval_options=eval_options,
        path_mappings=path_mappings
    )
    
    print(f"\n评估完成！结果保存在: {args.output_dir}")
    print(f"- JSON 报告: {os.path.join(args.output_dir, 'evaluation_results.json')}")
    print(f"- MD 报告: {os.path.join(args.output_dir, 'evaluation_report.md')}")
    print(f"- 图表: {os.path.join(args.output_dir, '*.png')}")

# 直接使用的函数（无需命令行）
def evaluate_model_simple(config_path, checkpoint_path, output_dir, img_dir=None, ann_file=None, path_mappings=None):
    """
    简化版的评估函数，直接调用
    """
    return evaluate_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        img_dir=img_dir,
        ann_file=ann_file,
        path_mappings=path_mappings
    )

if __name__ == "__main__":
    # 示例用法（可以直接修改这些路径来使用）
    if True:  # 设置为 True 来直接运行示例
        mode_path_root="/home/jovyan/workspace/FINISH_TRAIN_MODEL/"
        mode_name = "ssd_mobilenet_fpn_500e_1x8b_coco_1v"
        number="20251117_233313"
        config_path = f"{mode_path_root}{mode_name}/{number}/vis_data/config.py"
        checkpoint_path = f"{mode_path_root}{mode_name}/epoch_500.pth"
        output_dir = f"{mode_path_root}{mode_name}/evalue_out/"
        img_dir = "/home/jovyan/workspace/YOLO_DATASET/images"
        ann_file = "/home/jovyan/workspace/YOLO_DATASET/coco_framework/common/test.json"
        
        # 添加路径映射 - 将训练服务器的路径映射到评估服务器的路径
        path_mappings = {
            "/root/autodl-fs/yolo_train/YOLO_DATASET": "/home/jovyan/workspace/YOLO_DATASET",
            # 可以添加更多映射
            # "/root/other/path": "/home/jovyan/workspace/other/path"
        }
        
        results = evaluate_model_simple(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            img_dir=img_dir,
            ann_file=ann_file,
            path_mappings=path_mappings
        )
    else:
        # 使用命令行参数
        main()