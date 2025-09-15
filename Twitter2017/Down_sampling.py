import json
import random

# 输入输出文件路径
input_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/experiment/part.json"   # 你的原始文件
output_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/experiment/300.json"  # 保存下采样后的文件

# 设置随机种子保证复现
random.seed(42)

# 读取数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 按标签分类
label_to_samples = {"positive": [], "negative": [], "neutral": []}

for item in data:
    conversations = item["conversations"]
    # gpt 的回答作为标签
    for conv in conversations:
        if conv["from"] == "gpt":
            label = conv["value"].strip().lower()
            if label in label_to_samples:
                label_to_samples[label].append(item)

# 每类下采样 200 条
target_num = 100
balanced_data = []

for label, samples in label_to_samples.items():
    if len(samples) >= target_num:
        balanced_samples = random.sample(samples, target_num)
    else:
        raise ValueError(f"标签 {label} 样本不足 {target_num} 条，只有 {len(samples)} 条")
    balanced_data.extend(balanced_samples)

# 打乱顺序
random.shuffle(balanced_data)

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(balanced_data, f, ensure_ascii=False, indent=4)

print(f"✅ 已保存下采样后的数据，共 {len(balanced_data)} 条，每类 {target_num} 条")
