import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# 定义预处理函数来移除停用词
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

# 设置nltk数据路径
nltk.data.path.append('/home/kmyh/data/wd/nltk_data')

# 加载本地的英文停用词
stop_words = set(stopwords.words('english'))

# 输入 JSON 文件路径
json_path = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/train_output.json"

# 读取 JSON 文件并构建文本列表
texts = []
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        if "text" in item and item["text"].strip():
            texts.append(item["text"].strip())

# 预处理文本，移除停用词
processed_texts = [remove_stopwords(text) for text in texts]

# 检查文本列表
print("示例文本：", processed_texts[:5])

# 指定本地路径
embedding_model_path = "/home/kmyh/data/wd/code/data/models/all-MiniLM-L6-v2/"

# 创建 BERTopic 模型并使用本地模型
topic_model = BERTopic(embedding_model=embedding_model_path)

# 训练模型
topics, probabilities = topic_model.fit_transform(processed_texts)

# 获取所有主题
all_topics = topic_model.get_topics()

# 提取每个主题的前两个关键词（忽略 -1 主题）
topic_keywords = {}
for topic_id, words in all_topics.items():
    if topic_id != -1:  # 跳过 Topic -1
        top_keywords = [word for word, _ in words[:2]]  # 取前两个关键词
        topic_keywords[topic_id] = top_keywords

# 显示结果
for topic_id, keywords in topic_keywords.items():
    print(f"Topic {topic_id}: {keywords}")

# 保存结果到文件
df = pd.DataFrame(
    [{"Topic": tid, "Top Keywords": ", ".join(words)} for tid, words in topic_keywords.items()]
)
df.to_csv("/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/Twitter_topics_keywords.csv", index=False, encoding="utf-8")


