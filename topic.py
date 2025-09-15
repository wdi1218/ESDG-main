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

# 读取文本数据
mvsa_txt_path = "/home/kmyh/data/wd/code/data/datasets/mvsa/mvsa_1.txt"
yelp_txt_path = "/home/kmyh/data/wd/code/data/datasets/yelp/yelp_2.txt"
txt_path = mvsa_txt_path
texts = []

# 读取 TXT 文件并构建文本列表
with open(txt_path, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():  # 跳过空行
            parts = line.strip().split("\t")
            text = parts[1]  # 获取文本内容（假设文本在第3列）
            texts.append(text)

# 预处理文本，移除停用词
processed_texts = [remove_stopwords(text) for text in texts]

# 检查文本列表
print(processed_texts[:5])  # 打印前5个文本

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
        # 获取每个主题的前两个关键词
        top_keywords = [word for word, _ in words[:2]]  # 提取前两个关键词
        topic_keywords[topic_id] = top_keywords

# 显示结果
for topic_id, keywords in topic_keywords.items():
    print(f"Topic {topic_id}: {keywords}")

# 如果你想保存到文件，可以使用以下代码将结果保存为 CSV 文件
df = pd.DataFrame(list(topic_keywords.items()), columns=["Topic", "Top Keywords"])
df.to_csv("/home/kmyh/data/wd/code/data/datasets/topic_keyword/mvsa_topics_keywords.txt", index=False)

