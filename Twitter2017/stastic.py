import csv
from collections import Counter

# 输入文件路径
file_path = "/home/kmyh/data/wd/code/data/datasets/IJCAI2019_data/twitter2017/train.tsv"

def count_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        next(tsv_reader)  # 跳过表头
        for row in tsv_reader:
            if len(row) < 2:
                continue
            label = row[1].strip()  # 第二列是 label
            labels.append(label)
    return Counter(labels)

label_count = count_labels(file_path)

# 转换为更易读的标签分布
label_map = {"0": "消极", "1": "中性", "2": "积极"}
for k, v in label_count.items():
    print(f"标签 {k} ({label_map.get(k, '未知')}): {v} 条")
