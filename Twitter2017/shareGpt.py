import json
from collections import Counter

# 输入输出路径
input_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/test_output.json"
output_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/imagecaption_Tw_test.json"

# 固定指令
instruction = (
    "You are a sentiment analysis expert. You need to read the input image caption and the given text "
    "and then determine its sentiment label. Sentiment labels include positive, negative, and neutral. "
    "Make sure to analyze both the image caption and the text to determine the sentiment."
)

# 读取原始数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []
label_counter = Counter()  # 统计标签数量

for item in data:
    image_caption = item.get("image_caption") or item.get("caption") or ""
    text = item.get("text", "")
    label = item.get("value", "").lower()  # 统一小写

    # 更新计数器
    label_counter[label] += 1

    human_value = f"{instruction} \nImage caption: {image_caption}\nText: {text}"

    converted.append({
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": label}
        ]
    })

# 保存转换后的 ShareGPT 格式
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

# 打印标签统计
print("✅ 转换完成，保存到:", output_file)
print("样本标签数量统计：")
for lbl, count in label_counter.items():
    print(f"{lbl}: {count}")
