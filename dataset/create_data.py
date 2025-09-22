import argparse
import json
import os
import random

# 1. 命令行参数配置（保持原有功能，新增“类别数量范围”参数）
parser = argparse.ArgumentParser(description='生成“Nx class1, Nx class2; 场景”格式的prompt数据集')
parser.add_argument('--output_directory', type=str, default='dataset',
                    help='数据集保存目录')
parser.add_argument('--no_scene_percent', type=float, default=0.5,
                    help='无场景样本的比例（0~1）')
parser.add_argument('--N_samples', type=int, default=200,
                    help='总样本数量')
parser.add_argument('--max_classes_per_sample', type=int, default=1,
                    help='每个样本的最大类别数（建议1~2，避免过多类别混乱）')
parser.add_argument('--min_count', type=int, default=1,
                    help='每个类别的最小数量（如2）')
parser.add_argument('--max_count', type=int, default=8,
                    help='每个类别的最大数量（如5）')
args = parser.parse_args()


# 2. 基础配置（复用COCO目标映射，适配你的类别需求）
coco_objs = {
    "person": [{"id": 0, "name": "person"}], 
    "vehicle": [{"id": 1, "name": "bicycle"}, {"id": 2, "name": "car"}, {"id": 3, "name": "motorcycle"}, {"id": 4, "name": "airplane"}, {"id": 5, "name": "bus"}, {"id": 6, "name": "train"}, {"id": 7, "name": "truck"}, {"id": 8, "name": "boat"}], 
    "outdoor": [{"id": 9, "name": "traffic light"}, {"id": 10, "name": "fire hydrant"}, {"id": 11, "name": "stop sign"}, {"id": 12, "name": "parking meter"}, {"id": 13, "name": "bench"}], 
    "animal": [{"id": 14, "name": "bird"}, {"id": 15, "name": "cat"}, {"id": 16, "name": "dog"}, {"id": 17, "name": "horse"}, {"id": 18, "name": "sheep"}, {"id": 19, "name": "cow"}, {"id": 20, "name": "elephant"}, {"id": 21, "name": "bear"}, {"id": 22, "name": "zebra"}, {"id": 23, "name": "giraffe"}],
    "accessory": [{"id": 24, "name": "backpack"}, {"id": 25, "name": "umbrella"}, {"id": 26, "name": "handbag"}, {"id": 27, "name": "tie"}, {"id": 28, "name": "suitcase"}], 
    "sports": [{"id": 29, "name": "frisbee"}, {"id": 30, "name": "skis"}, {"id": 31, "name": "snowboard"}, {"id": 32, "name": "sports ball"}, {"id": 33, "name": "kite"}, {"id": 34, "name": "baseball bat"}, {"id": 35, "name": "baseball glove"}, {"id": 36, "name": "skateboard"}, {"id": 37, "name": "surfboard"}, {"id": 38, "name": "tennis racket"}], 
    "kitchen": [{"id": 39, "name": "bottle"}, {"id": 40, "name": "wine glass"}, {"id": 41, "name": "cup"}, {"id": 42, "name": "fork"}, {"id": 43, "name": "knife"}, {"id": 44, "name": "spoon"}, {"id": 45, "name": "bowl"}], 
    "food": [{"id": 46, "name": "banana"}, {"id": 47, "name": "apple"}, {"id": 48, "name": "sandwich"}, {"id": 49, "name": "orange"}, {"id": 50, "name": "broccoli"}, {"id": 51, "name": "carrot"}, {"id": 52, "name": "hot dog"}, {"id": 53, "name": "pizza"}, {"id": 54, "name": "donut"}, {"id": 55, "name": "cake"}], 
    "furniture": [{"id": 56, "name": "chair"}, {"id": 57, "name": "couch"}, {"id": 58, "name": "potted plant"}, {"id": 59, "name": "bed"}, {"id": 60, "name": "dining table"}, {"id": 61, "name": "toilet"}], 
    "electronic": [{"id": 62, "name": "tv"}, {"id": 63, "name": "laptop"}, {"id": 64, "name": "mouse"}, {"id": 65, "name": "remote"}, {"id": 66, "name": "keyboard"}, {"id": 67, "name": "cell phone"}], 
    "appliance": [{"id": 68, "name": "microwave"}, {"id": 69, "name": "oven"}, {"id": 70, "name": "toaster"}, {"id": 71, "name": "sink"}, {"id": 72, "name": "refrigerator"}], 
    "indoor": [{"id": 73, "name": "book"}, {"id": 74, "name": "clock"}, {"id": 75, "name": "vase"}, {"id": 76, "name": "scissors"}, {"id": 77, "name": "teddy bear"}, {"id": 78, "name": "hair drier"}, {"id": 79, "name": "toothbrush"}]
}
# 筛选你关注的核心类别（可根据需求增删）
target_object_list = ['cat', 'dog', 'bird', 'car', 'airplane', 'cup', 'apple', 'donut']
# COCO别名映射（保持原有逻辑，确保类别名正确）
coco_object_name_dict = {'ball': 'sports ball', 'glove': 'baseball glove', 'phone': 'cell phone'}
# 场景列表（适配你的格式，可扩展）
scenes = ['on green grass', 'on the road', 'on a wooden desk', 'in a white room']
# 生成数量范围（从参数读取，如2~5）
count_range = list(range(args.min_count, args.max_count + 1))


# 3. 辅助函数：生成随机的“类别-数量”组合（支持多类别）
def get_random_class_count_pairs(object_list, max_classes=2, count_range=[2,3,4,5]):
    """
    生成1~max_classes个“类别-数量”对，避免重复类别
    返回：(class_count_list, class_counts_dict)
    - class_count_list: 如[("cat",3), ("dog",2)]（用于拼接prompt）
    - class_counts_dict: 如{"cat":3, "dog":2}（直接给你的make_oct_embeddings用）
    """
    # 随机选1~max_classes个不重复的类别
    num_classes = random.randint(1, max_classes)
    selected_classes = random.sample(object_list, num_classes)  # 避免重复
    
    class_count_list = []
    class_counts_dict = {}
    for cls in selected_classes:
        count = random.choice(count_range)  # 每个类别随机分配数量
        # 处理COCO别名映射（确保类别名与COCO一致）
        coco_cls = cls if cls not in coco_object_name_dict else coco_object_name_dict[cls]
        class_count_list.append((cls, count))  # prompt用单数类别（如cat）
        class_counts_dict[cls] = count  # class_counts用单数类别（匹配你的需求）
    return class_count_list, class_counts_dict


# 4. 核心逻辑：生成prompt+class_counts+元信息
if __name__ == "__main__":
    # 初始化变量
    output_dir = args.output_directory
    no_scene_percent = args.no_scene_percent
    N_samples = args.N_samples
    max_classes = args.max_classes_per_sample
    
    # 构建COCO目标ID映射（复用原有逻辑，用于元信息记录）
    all_coco_objs = [item for sublist in coco_objs.values() for item in sublist]
    obj_to_id = {x["name"]: x["id"] for x in all_coco_objs}
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "data_with_class_counts.json")
    
    prompts_with_meta = []  # 存储每个样本的完整信息（含class_counts）
    
    for i in range(N_samples):
        # 步骤1：生成随机“类别-数量”对和class_counts
        class_count_pairs, class_counts = get_random_class_count_pairs(
            object_list=target_object_list,
            max_classes=max_classes,
            count_range=count_range
        )
        
        # 步骤2：拼接“Nx class1, Nx class2”部分（核心格式适配）
        class_part = ", ".join([f"{count}x {cls}" for cls, count in class_count_pairs])
        
        # 步骤3：处理场景（用“;”分隔，无场景则省略）
        scene = ""
        if random.random() > no_scene_percent:  # 有场景的概率=1-no_scene_percent
            scene = random.choice(scenes)
            prompt = f"{class_part}; {scene}"
        else:
            prompt = class_part  # 无场景时仅保留“Nx class”部分
        
        # 步骤4：记录COCO目标ID（用于元信息追溯，可选）
        coco_ids = {}
        for cls in class_counts.keys():
            coco_cls = cls if cls not in coco_object_name_dict else coco_object_name_dict[cls]
            coco_ids[cls] = obj_to_id.get(coco_cls, -1)  # -1表示未匹配到COCO ID
        
        # 步骤5：生成随机种子（用于复现图像生成）
        seed = random.randint(0, 10**6)
        
        # 步骤6：组装完整样本信息（含class_counts）
        sample = {
            "prompt": prompt,                  # 你的目标格式：如“3x cat, 2x dog; on green grass”
            "class_counts": class_counts,      # 直接可用的字典：如{"cat":3, "dog":2}
            "coco_object_ids": coco_ids,       # COCO ID元信息（可选，便于后续扩展）
            "scene": scene,                    # 场景描述（空表示无场景）
            "seed": seed,                      # 随机种子（复现生成结果）
            "sample_idx": i + 1                # 样本序号（便于定位）
        }
        prompts_with_meta.append(sample)
    
    # 步骤7：保存到JSON（一次性写入，提升效率）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prompts_with_meta, f, indent=4)
    
    print(f"生成完成！共{len(prompts_with_meta)}条样本，保存路径：{json_path}")
    # 打印1条示例，验证格式
    print("\n示例样本：")
    print(json.dumps(prompts_with_meta[0], indent=4))