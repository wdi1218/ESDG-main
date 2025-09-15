import csv
import json

# 输入 TSV 文件路径
tsv_file = "/home/kmyh/data/wd/code/data/datasets/IJCAI2019_data/twitter2017/train.tsv"
# 图片目录
image_dir = "/home/kmyh/data/wd/code/data/datasets/IJCAI2019_data/twitter2017_images"
# 输出 JSON 文件
output_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/train.json"

# 标签映射
label_map = {"0": "negative", "1": "neutral", "2": "positive"}

data_out = []

with open(tsv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)  # 跳过表头
    for row in reader:
        if len(row) < 4:
            continue
        idx, label, image_id, text1, *text2 = row
        label = label.strip()
        text1 = text1.strip()
        text2 = " ".join(text2).strip()  # 防止多列字符串
        text = f"{text1} {text2}".strip()

        # 构建 sharegpt 格式
        entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"Now you are a sentiment analysis expert. You need to read the input image and text data and then determine its sentiment label. Sentiment labels include positive, negative and neutral.<image>{text}"
                },
                {
                    "from": "gpt",
                    "value": label_map.get(label, "neutral")
                }
            ],
            "images": [
                f"{image_dir}/{image_id}"
            ]
        }
        data_out.append(entry)

# 保存 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_out, f, indent=2, ensure_ascii=False)

print(f"转换完成，共 {len(data_out)} 条数据，已保存到 {output_file}")
