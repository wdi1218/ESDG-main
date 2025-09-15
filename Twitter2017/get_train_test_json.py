import json
import random
from collections import defaultdict

# 输入输出路径
input_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/experiment/output.json"
output_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/experiment/part.json"
random.seed(42)  # 固定随机种子
# 读取数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"原始样本总数: {len(data)}")

# 按标签分类存放
label_to_items = defaultdict(list)
for item in data:
    for conv in item["conversations"]:
        if conv["from"] == "gpt":
            label = conv["value"].strip().lower()
            label_to_items[label].append(item)
            break  # 每条只取一个标签

# 统计各标签数量
for label, items in label_to_items.items():
    print(f"{label}: {len(items)}")

# 找出最小的类别数
min_count = min(len(items) for items in label_to_items.values())
print(f"最小类别样本数: {min_count}")

# 按 min_count 采样
balanced_data = []
for label, items in label_to_items.items():
    if len(items) > min_count:
        balanced_data.extend(random.sample(items, min_count))
    else:
        balanced_data.extend(items)  # 如果刚好是最小类，取全量

print(f"下采样后的样本总数: {len(balanced_data)}")

# 保存文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(balanced_data, f, ensure_ascii=False, indent=2)

print("已保存到:", output_file)
