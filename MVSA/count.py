import pandas as pd

# 文件路径
label_file_path = 'MVSA/mvsa_labels.txt'

# 读取 mvsa_label.txt 文件
labels_df = pd.read_csv(label_file_path, sep='\t', header=None, names=['ID', 'Label'])

# 统计每个标签的数量
label_counts = labels_df['Label'].value_counts()

# 输出结果
print(label_counts)

# 如果需要保存结果到文件
output_file_path = 'MVSA/label_counts.txt'
label_counts.to_csv(output_file_path, sep='\t', header=False)

print(f"Label counts saved to {output_file_path}")