import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 本地模型路径
model_path = '/home/kmyh/data/wd/code/data/models/paraphrase-MiniLM-L6-v2/'

# 加载本地的 Sentence-BERT 模型
model = SentenceTransformer(model_path)
# /home/kmyh/data/wd/code/data/datasets/RAG_TIQU/mvsa_test.json
# /home/kmyh/data/wd/code/data/datasets/RAG_TIQU/yelp_train.json
# /home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/train_output.json
# 读取 JSON 文件
mvsa_json_path = "/home/kmyh/data/wd/code/data/datasets/RAG_TIQU/mvsa_test.json"
yelp_json_path = "/home/kmyh/data/wd/code/data/datasets/RAG_TIQU/yelp_train.json"
twitter_json_path = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/twitter2017/data/test_output.json"

def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

mvsa_data = load_json_data(mvsa_json_path)
yelp_data = load_json_data(yelp_json_path)
twitter_data = load_json_data(twitter_json_path)

# 源域和目标域
source_data = yelp_data
target_data = mvsa_data

# 提取文本
source_sentences = [entry['text'] for entry in source_data]
target_sentences = [entry['text'] for entry in target_data]

# 计算嵌入
source_embeddings = model.encode(source_sentences)
target_embeddings = model.encode(target_sentences)

# 相似度矩阵（双向+平均）
similarities_target_to_source = cosine_similarity(target_embeddings, source_embeddings)
similarities_source_to_target = cosine_similarity(source_embeddings, target_embeddings)
similarities = (similarities_target_to_source + similarities_source_to_target.T) / 2

# 标签类别（根据实际数据集的标签修改，如 "positive"/"neutral"/"negative"）
labels = ["positive", "neutral", "negative"]

# 先将源域样本按标签分组
source_by_label = {label: [] for label in labels}
for idx, entry in enumerate(source_data):
    label = entry["value"]
    if label in source_by_label:
        source_by_label[label].append(idx)

# 每个目标样本取的示例数量
n_examples = 1  # 可以改成5、6等 ######################################################
per_class = n_examples // len(labels)  # 每类的基础数量
remainder = n_examples % len(labels)   # 余数分配

output_data = []

for i, target_entry in enumerate(target_data):
    target_text = target_entry['text']
    target_value = target_entry['value']
    target_image_caption = target_entry['image_caption']

    similarity_scores = similarities[i]
    selected_indices = []

    # 按类别取样
    for li, label in enumerate(labels):
        k_class = per_class + (1 if li < remainder else 0)  # 分配余数
        candidate_indices = source_by_label[label]
        if not candidate_indices:
            continue

        # 当前类别样本的相似度排序
        candidate_scores = [(idx, similarity_scores[idx]) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # 选前 k_class 个
        selected_indices.extend([idx for idx, _ in candidate_scores[:k_class]])

    # 如果总数不足，补齐
    if len(selected_indices) < n_examples:
        all_sorted = np.argsort(similarity_scores)[::-1]
        for idx in all_sorted:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= n_examples:
                break

    # 构造示例，⚠️这里加入了标签
    top_n_examples = []
    for idx in selected_indices:
        example_text = source_data[idx]['text']
        example_image_caption = source_data[idx]['image_caption']
        example_value = source_data[idx]['value']   # 取标签
        formatted_example = (
            f"\nImage caption: {example_image_caption}"
            f"\nText: {example_text}"
            f"\nLabel: {example_value}"
        )
        top_n_examples.append(formatted_example)

    output_entry = {
        "image_caption": target_image_caption,
        "text": target_text,
        "value": target_value,
    }
    for j, example in enumerate(top_n_examples):
        output_entry[f"example{j+1}"] = example

    output_data.append(output_entry)

# 保存结果
output_json_path = "/home/kmyh/data/wd/code/llm/LLaMA-Factory-829/data_solve/data/yeTr_mvTe_examples_1.json"
with open(output_json_path, "w", encoding="utf-8") as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)

print(f"Processed data saved to {output_json_path}")
