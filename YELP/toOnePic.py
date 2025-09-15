import json

# 文件路径
input_file_path = 'YELP/test/New York_test.json'
output_file_path = 'test_ny.txt'

# 函数：将 Rating 转换为情感标签
def convert_rating(rating):
    if rating in [1, 2]:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    elif rating in [4, 5]:
        return 'positive'

# 初始化计数器
total_count = 0
positive_count = 0
neutral_count = 0
negative_count = 0


# 打开文件并逐行读取
output_data = []
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            rating = convert_rating(data['Rating'])
            text = data['Text'].split('|||')[0]  # 只读取 '|||' 前面的部分(第一幅图)
            user_id = data['UserId']
            first_photo_id = data['Photos'][0]['_id'] if data['Photos'] else None
            output_data.append([user_id, first_photo_id, text, rating])

            # 更新计数器
            total_count += 1
            if rating == 'positive':
                positive_count += 1
            elif rating == 'neutral':
                neutral_count += 1
            elif rating == 'negative':
                negative_count += 1

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 写入到新的文本文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("u_id\tp_id\ttext\tlabel\n")
    for record in output_data:
        file.write(f"{record[0]}\t{record[1]}\t{record[2]}\t{record[3]}\n")
        
# 打印样本总数和每个情感标签的数量
print(f"Total samples: {total_count}")
print(f"Positive samples: {positive_count}")
print(f"Neutral samples: {neutral_count}")
print(f"Negative samples: {negative_count}")

print(f"Extracted data saved to {output_file_path}")