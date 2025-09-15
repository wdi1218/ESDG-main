import json

input_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/test_caption_prompt.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

bad_samples = []
for idx, item in enumerate(data):
    # 统计 human 对话里 <image> 出现次数
    image_tokens = 0
    for conv in item["conversations"]:
        if conv["from"] == "human":
            image_tokens += conv["value"].count("<image>")

    # 对比 images 数量
    num_images = len(item.get("images", []))
    if image_tokens != num_images:
        bad_samples.append((idx, image_tokens, num_images))

print(f"总样本数: {len(data)}")
print(f"有问题的样本数: {len(bad_samples)}")
if bad_samples:
    print("前几个问题样本:", bad_samples[:10])
