import pandas as pd

# 从文件读取数据
file_path = 'MVSA/labelResultAll.txt'
data = pd.read_csv(file_path, sep='\t', header=None)

# 确保数据的格式正确
def split_columns(row):
    cols = [row[0]]
    for i in range(1, 4):
        text, image = row[i].split(',')
        cols.append(text)
        cols.append(image)
    return pd.Series(cols)

# 拆分列并添加列名
data = data.apply(split_columns, axis=1)
data.columns = ["ID", "text1", "image1", "text2", "image2", "text3", "image3"]

def determine_label(text, image):
    if text == "neutral":
        return image
    if image == "neutral":
        return text
    if text == image:
        return text
    return None

def process_sample(row):
    labels = []
    for i in range(1, 4):
        text = row[f"text{i}"]
        image = row[f"image{i}"]
        label = determine_label(text, image)
        if label:
            labels.append(label)
    return labels

def final_label(labels):
    # 应用投票机制
    if len(labels) >= 2:
        label_counts = pd.Series(labels).value_counts()
        if label_counts.iloc[0] >= 2:
            return label_counts.index[0]
    return None

processed_data = []

for _, row in data.iterrows():
    labels = process_sample(row)
    if labels:
        final = final_label(labels)
        if final:
            processed_data.append([row["ID"], final])

# 转换为DataFrame
final_df = pd.DataFrame(processed_data, columns=["ID", "Final_Label"])

# 输出结果到新的文件
output_path = 'MVSA/mvsa_labels.txt'
final_df.to_csv(output_path, index=False, sep='\t')

print(f"Processed data saved to {output_path}")