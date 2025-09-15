import json

input_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/train.json"
output_file = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/train_caption_with_text.json"

# image caption 前缀
prompt_prefix = (
    "You are an image captioning expert. Please describe the given image in one clear and concise sentence, "
    "focusing only on what is visible in the picture.<image>"
)

# 新的 prompt
new_prompt = "You are an image captioning expert. Please describe the given image in one clear and concise sentence, focusing only on what is visible in the picture.<image>"

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历每个条目
for item in data:
    for conv in item.get("conversations", []):
        if conv.get("from") == "human":
            text = conv.get("value", "")
            if "<image>" in text:
                # 提取 <image> 后面的原文本
                after_image = text.split("<image>", 1)[1]
                # 构建新的 value
                conv["value"] = f"{new_prompt} Reference text:{after_image.strip()}"

# 保存修改后的 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON 文件已更新完成！")