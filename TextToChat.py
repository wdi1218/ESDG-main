import json

# 读取原始数据
input_json_path = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/data/yeTr_mvTe_examples_3.json"
output_json_path = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/Experiment_YeMv/yeTr_mvTe_only_examples_3.json"


# 读取原始 JSON 数据
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


data = load_json_data(input_json_path)

# 构建对话数据
conversations = []

# 遍历每个条目
for entry in data:
    # 获取目标文本的图片描述、文本和情感标签
    image_caption = entry["image_caption"]
    text = entry["text"]
    value = entry["value"]

    # 格式化为 prompt（用户输入）
    prompt = (
        f"You are a sentiment analysis expert."
        f"You need to read the input image caption and the given text and then determine its sentiment label. Sentiment labels include positive, negative, and neutral. Make sure to analyze both the image caption and the text to determine the sentiment. "
        f"Below are some examples:\n\n"
    )

    #添加示例到 prompt 中
    for i in range(1, 4):
        example_key = f"example{i}"
        prompt += f"Example{i}: {entry[example_key]}\n"

    # The dataset includes a variety of topics centered around geographic locations (e.g., Toronto, Montreal, Calgary, Vancouver), events (e.g., concerts, festivals, weddings), lifestyle (e.g., vegan food, fitness, art), and trending social media topics (e.g., popular hashtags, celebrities, fan engagement). The data reflects diverse trends in Canadian culture, entertainment, politics, and local events.
    # The dataset centers on consumer reviews of various restaurants and food experiences, highlighting popular cuisines like sushi, pizza, ramen, and vegan options. It captures user sentiments on service, food quality, pricing, and ambiance, with a strong focus on local dining, reservations, and delivery services. Trends reflect both positive and negative feedback on dining experiences, including mentions of specific dishes, locations, and customer service quality.
    # The dataset covers diverse topics including sports leagues, music concerts, film, fashion, and celebrity culture, alongside politics and social issues. It highlights public attention trends around entertainment events, famous personalities, and global news. \nImage caption: A man holding a drumstick is performing on stage.

    prompt += f"Now, please analyze the following example."
    # prompt += "The dataset includes a variety of topics centered around geographic locations (e.g., Toronto, Montreal, Calgary, Vancouver), events (e.g., concerts, festivals, weddings), lifestyle (e.g., vegan food, fitness, art), and trending social media topics (e.g., popular hashtags, celebrities, fan engagement). The data reflects diverse trends in Canadian culture, entertainment, politics, and local events. \n"
    prompt += f"Image caption: {image_caption}\nText: {text}"

    # 创建对话格式
    conversation = {
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": value  # 模型输出（情感标签）
            }
        ]
    }

    conversations.append(conversation)

# 保存为新的 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as output_file:
    json.dump(conversations, output_file, ensure_ascii=False, indent=4)

print(f"Processed data saved to {output_json_path}")