import os
import json
import argparse
import numpy as np
import pandas as pd
import PIL.Image
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Union
import yaml
import torch
from tqdm import tqdm

from oct_ccg.pipeline.run_oct_ccg import read_yaml, set_seed, load_dataset, parse_class_counts_from_prompt, init_pipeline, run_oct_ccg_pipeline


class OCTCCGEvaluator:
    def __init__(self, output_dir: str):
        """
        初始化OCT-CCG评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化YOLO模型用于物体检测
        self.model = YOLO('yolov9e.pt')
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # 存储评估结果
        self.results = []
        
    def evaluate_generated_image(self, image_path: str, expected_class_counts: Dict[str, int]) -> Dict:
        """
        评估生成的图像
        
        Args:
            image_path: 图像路径
            expected_class_counts: 期望的类别计数
            
        Returns:
            包含评估结果的字典
        """
        # 加载图像
        pil_image = PIL.Image.open(image_path)
        
        # 使用YOLO进行物体检测
        result = self.model(pil_image)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # 分析检测结果
        detected_counts = {}
        class_results = {}
        
        # 统计每个类别的检测数量
        for class_name in expected_class_counts.keys():
            # 计算该类别的检测数量
            class_detections = detections[detections.data['class_name'] == class_name]
            detected_count = len(class_detections)
            detected_counts[class_name] = detected_count
            
            # 计算该类别的准确率
            expected_count = expected_class_counts[class_name]
            is_correct = detected_count == expected_count
            
            # 计算绝对误差和相对误差
            abs_error = abs(detected_count - expected_count)
            rel_error = abs_error / max(expected_count, 1)  # 避免除以0
            
            class_results[class_name] = {
                "expected_count": expected_count,
                "detected_count": detected_count,
                "is_correct": is_correct,
                "absolute_error": abs_error,
                "relative_error": rel_error
            }
        
        # 计算整体指标
        total_expected = sum(expected_class_counts.values())
        total_detected = sum(detected_counts.values())
        total_abs_error = sum(abs(detected_counts[c] - expected_class_counts[c]) for c in expected_class_counts.keys())
        total_rel_error = total_abs_error / max(total_expected, 1)
        
        # 计算准确率（所有类别都正确才算正确）
        all_correct = all(class_results[c]["is_correct"] for c in expected_class_counts.keys())
        
        # 创建带标注的图像
        annotated_frame = self._create_annotated_image(pil_image, detections)
        annotated_image_path = os.path.join(self.output_dir, f"annotated_{os.path.basename(image_path)}")
        annotated_frame.save(annotated_image_path)
        
        # 返回评估结果
        return {
            "class_results": class_results,
            "total_expected": total_expected,
            "total_detected": total_detected,
            "total_absolute_error": total_abs_error,
            "total_relative_error": total_rel_error,
            "all_correct": all_correct,
            "annotated_image_path": annotated_image_path,
            "image_path": image_path
        }
    
    def _create_annotated_image(self, pil_image: PIL.Image, detections):
        """
        创建带标注的图像
        
        Args:
            pil_image: PIL图像对象
            detections: 检测结果
            
        Returns:
            带标注的图像
        """
        annotated_frame = pil_image.copy()
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        return annotated_frame
    
    def evaluate_from_metadata(self, metadata_dir: str, images_dir: Optional[str] = None) -> List[Dict]:
        """
        从元数据文件评估已生成的图像
        
        Args:
            metadata_dir: 元数据文件所在目录
            images_dir: 图像所在目录，如果为None则使用与元数据相同的目录
            
        Returns:
            评估结果列表
        """
        if images_dir is None:
            images_dir = metadata_dir
            
        # 获取所有元数据文件
        metadata_files = [f for f in os.listdir(metadata_dir) if f.startswith("metadata_") and f.endswith(".json")]
        
        results = []
        for metadata_file in tqdm(metadata_files, desc="评估图像"):
            metadata_path = os.path.join(metadata_dir, metadata_file)
            
            # 加载元数据
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 获取图像路径
            image_name = metadata_file.replace("metadata_", "output_").replace(".json", ".png")
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                continue
            
            # 评估图像
            try:
                class_counts = metadata.get("class_counts", parse_class_counts_from_prompt(metadata["prompt"]))
                result = self.evaluate_generated_image(image_path, class_counts)
                
                # 添加元数据信息
                result["prompt"] = metadata["prompt"]
                result["seed"] = metadata.get("seed")
                result["metadata_path"] = metadata_path
                
                results.append(result)
            except Exception as e:
                print(f"评估图像失败 {image_path}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def evaluate_from_image_names(self, images_dir: str) -> List[Dict]:
        """
        从图像文件名评估已生成的图像（根据run_oct_ccg.py的保存逻辑）
        
        Args:
            images_dir: 图像所在目录
            
        Returns:
            评估结果列表
        """
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
        
        results = []
        for image_file in tqdm(image_files, desc="评估图像"):
            image_path = os.path.join(images_dir, image_file)
            
            try:
                # 从文件名解析期望计数和类别
                class_counts = self._parse_class_counts_from_filename(image_file)
                
                # 评估图像
                result = self.evaluate_generated_image(image_path, class_counts)
                
                # 添加元数据信息
                result["filename"] = image_file
                
                results.append(result)
            except Exception as e:
                print(f"评估图像失败 {image_path}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def _parse_class_counts_from_filename(self, filename: str) -> Dict[str, int]:
        """
        从文件名解析类别计数（根据run_oct_ccg.py的保存逻辑）
        
        Args:
            filename: 图像文件名
            
        Returns:
            类别计数字典
        """
        # 去除文件扩展名
        base_name = os.path.splitext(filename)[0]
        
        # 分离类别部分和索引/种子部分
        if "_idx=" in base_name:
            class_part = base_name.split("_idx=")[0]
        elif "_seed=" in base_name:
            class_part = base_name.split("_seed=")[0]
        else:
            # 如果没有索引或种子部分，假设整个文件名都是类别部分
            class_part = base_name
        
        # 解析类别部分
        class_counts = {}
        class_items = class_part.split("_")
        
        for item in class_items:
            if "_num=" in item:
                class_name, count_str = item.split("_num=")
                try:
                    count = int(count_str)
                    class_counts[class_name] = count
                except ValueError:
                    continue
        
        if not class_counts:
            raise ValueError(f"无法从文件名解析类别计数: {filename}")
        
        return class_counts
    
    def _analyze_image_name(self, image_name: str) -> Tuple[int, str]:
        """
        分析图像文件名，提取期望计数和类别（保留以兼容旧格式）
        
        Args:
            image_name: 图像文件名
            
        Returns:
            (期望计数, 类别名称)
        """
        # 首先尝试使用新的解析方法
        try:
            class_counts = self._parse_class_counts_from_filename(image_name)
            # 如果只有一个类别，返回该类别的计数和名称
            if len(class_counts) == 1:
                class_name, count = next(iter(class_counts.items()))
                return count, class_name
            else:
                # 如果有多个类别，返回第一个类别的计数和名称
                class_name, count = next(iter(class_counts.items()))
                return count, class_name
        except ValueError:
            # 如果新方法失败，尝试使用旧方法
            # 去除文件扩展名
            base_name = os.path.splitext(image_name)[0]
            
            # 尝试不同的分隔符
            for separator in ["__", "_", "-"]:
                if separator in base_name:
                    parts = base_name.split(separator)
                    if len(parts) >= 2:
                        try:
                            expected_count = int(parts[0])
                            class_name = parts[1]
                            return expected_count, class_name
                        except ValueError:
                            continue
            
            # 如果无法解析，尝试查找数字
            import re
            match = re.search(r'(\d+)', base_name)
            if match:
                expected_count = int(match.group(1))
                # 假设类别名称是文件名中数字后面的部分
                class_name = base_name[match.end():].strip("_- ")
                if class_name:
                    return expected_count, class_name
            
            raise ValueError(f"无法从图像文件名解析期望计数和类别: {image_name}")
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        计算整体评估指标
        
        Args:
            results: 评估结果列表
            
        Returns:
            包含整体指标的字典
        """
        if not results:
            return {}
        
        # 初始化统计变量
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r["all_correct"])
        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        # 按类别统计
        class_metrics = {}
        class_counts = {}
        
        for result in results:
            for class_name, class_result in result["class_results"].items():
                if class_name not in class_metrics:
                    class_metrics[class_name] = {
                        "total_samples": 0,
                        "correct_samples": 0,
                        "total_absolute_error": 0,
                        "total_relative_error": 0,
                        "total_expected": 0,
                        "total_detected": 0
                    }
                
                class_metrics[class_name]["total_samples"] += 1
                class_metrics[class_name]["correct_samples"] += 1 if class_result["is_correct"] else 0
                class_metrics[class_name]["total_absolute_error"] += class_result["absolute_error"]
                class_metrics[class_name]["total_relative_error"] += class_result["relative_error"]
                class_metrics[class_name]["total_expected"] += class_result["expected_count"]
                class_metrics[class_name]["total_detected"] += class_result["detected_count"]
                
                # 统计每个类别的样本数量
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        
        # 计算每个类别的指标
        for class_name, metrics in class_metrics.items():
            metrics["accuracy"] = metrics["correct_samples"] / metrics["total_samples"]
            metrics["mean_absolute_error"] = metrics["total_absolute_error"] / metrics["total_samples"]
            metrics["mean_relative_error"] = metrics["total_relative_error"] / metrics["total_samples"]
            metrics["counting_accuracy"] = 1 - (metrics["total_absolute_error"] / max(metrics["total_expected"], 1))
        
        # 计算整体指标
        total_absolute_error = sum(r["total_absolute_error"] for r in results)
        total_relative_error = sum(r["total_relative_error"] for r in results)
        total_expected = sum(r["total_expected"] for r in results)
        total_detected = sum(r["total_detected"] for r in results)
        
        overall_metrics = {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "accuracy": accuracy,
            "total_absolute_error": total_absolute_error,
            "mean_absolute_error": total_absolute_error / total_samples if total_samples > 0 else 0,
            "total_relative_error": total_relative_error,
            "mean_relative_error": total_relative_error / total_samples if total_samples > 0 else 0,
            "total_expected": total_expected,
            "total_detected": total_detected,
            "counting_accuracy": 1 - (total_absolute_error / max(total_expected, 1)),
            "class_metrics": class_metrics,
            "class_counts": class_counts
        }
        
        return overall_metrics
    
    def save_results(self, results: List[Dict], metrics: Dict, output_path: Optional[str] = None):
        """
        保存评估结果
        
        Args:
            results: 评估结果列表
            metrics: 整体指标
            output_path: 输出路径，None表示使用默认路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "evaluation_results.json")
        
        # 准备保存的数据
        save_data = {
            "metrics": metrics,
            "results": results
        }
        
        # 保存JSON文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        
        # 保存CSV文件（简化版结果）
        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        csv_data = []
        for result in results:
            row = {
                "image_path": result["image_path"],
                "all_correct": result["all_correct"],
                "total_expected": result["total_expected"],
                "total_detected": result["total_detected"],
                "total_absolute_error": result["total_absolute_error"],
                "total_relative_error": result["total_relative_error"]
            }
            
            # 添加prompt（如果存在）
            if "prompt" in result:
                row["prompt"] = result["prompt"]
            
            # 添加seed（如果存在）
            if "seed" in result:
                row["seed"] = result["seed"]
            
            # 添加class_name和expected_count（如果存在，用于兼容evaluation_script.py格式）
            if "class_name" in result and "expected_count" in result:
                row["class_name"] = result["class_name"]
                row["expected_count"] = result["expected_count"]
                row["detected_count"] = result["class_results"][result["class_name"]]["detected_count"]
                row["is_success"] = result["class_results"][result["class_name"]]["is_correct"]
            
            # 添加每个类别的结果
            for class_name, class_result in result["class_results"].items():
                row[f"{class_name}_expected"] = class_result["expected_count"]
                row[f"{class_name}_detected"] = class_result["detected_count"]
                row[f"{class_name}_is_correct"] = class_result["is_correct"]
                row[f"{class_name}_absolute_error"] = class_result["absolute_error"]
                row[f"{class_name}_relative_error"] = class_result["relative_error"]
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        # 保存指标摘要
        summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("OCT-CCG 评估结果摘要\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总样本数: {metrics['total_samples']}\n")
            f.write(f"正确样本数: {metrics['correct_samples']}\n")
            f.write(f"准确率: {metrics['accuracy']:.2%}\n")
            f.write(f"计数准确率: {metrics['counting_accuracy']:.2%}\n")
            f.write(f"平均绝对误差: {metrics['mean_absolute_error']:.4f}\n")
            f.write(f"平均相对误差: {metrics['mean_relative_error']:.2%}\n")
            f.write(f"总期望计数: {metrics['total_expected']}\n")
            f.write(f"总检测计数: {metrics['total_detected']}\n\n")
            
            f.write("各类别指标:\n")
            f.write("-" * 50 + "\n")
            for class_name, class_metrics in metrics["class_metrics"].items():
                f.write(f"{class_name} (样本数: {metrics['class_counts'][class_name]}):\n")
                f.write(f"  准确率: {class_metrics['accuracy']:.2%}\n")
                f.write(f"  计数准确率: {class_metrics['counting_accuracy']:.2%}\n")
                f.write(f"  平均绝对误差: {class_metrics['mean_absolute_error']:.4f}\n")
                f.write(f"  平均相对误差: {class_metrics['mean_relative_error']:.2%}\n")
                f.write(f"  总期望计数: {class_metrics['total_expected']}\n")
                f.write(f"  总检测计数: {class_metrics['total_detected']}\n\n")
        
        print(f"评估结果已保存到: {output_path}")
        print(f"CSV结果已保存到: {csv_path}")
        print(f"评估摘要已保存到: {summary_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="OCT-CCG评估脚本")
    parser.add_argument("--images_dir", type=str, required=True, help="图像所在目录")
    parser.add_argument("--metadata_dir", type=str, help="元数据文件所在目录，如果提供则从元数据读取期望计数")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建评估器
    evaluator = OCTCCGEvaluator(args.output_dir)
    
    # 评估图像
    if args.metadata_dir:
        # 从元数据文件评估
        results = evaluator.evaluate_from_metadata(args.metadata_dir, args.images_dir)
    else:
        # 从图像文件名评估（类似evaluation_script.py的方式）
        results = evaluator.evaluate_from_image_names(args.images_dir)
    
    # 计算整体指标
    metrics = evaluator.calculate_metrics(results)
    
    # 保存结果
    evaluator.save_results(results, metrics)
    
    # 打印摘要
    print("\n评估结果摘要:")
    print(f"总样本数: {metrics['total_samples']}")
    print(f"正确样本数: {metrics['correct_samples']}")
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"计数准确率: {metrics['counting_accuracy']:.2%}")
    print(f"平均绝对误差: {metrics['mean_absolute_error']:.4f}")
    print(f"平均相对误差: {metrics['mean_relative_error']:.2%}")


if __name__ == "__main__":
    main()