# pip install -U pandas scikit-learn 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# 读取数据集，数据集包含域名和标签（恶意或正常）
data = pd.read_csv("domain_data.csv")
data['domain'] = data['domain'].astype(str)

# 黑名单集合
domain_label_map = {}

# example:
# icb-online-intl.com,FALSE
# kobesportswear.date,FALSE

for index, row in data.iterrows():
    if row['label'] == False:
        domain_label_map[row['domain']] = row['label']

print("黑名单数量: ",len(domain_label_map))

# 提取特征：例如，域名长度、是否包含数字、是否包含特殊字符等
data['length'] = data['domain'].apply(len)
data['has_numbers'] = data['domain'].apply(lambda x: any(char.isdigit() for char in x))
data['has_special_chars'] = data['domain'].apply(lambda x: any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",.<>?/" for char in x))

# 划分数据集为训练集和测试集
X = data[['length', 'has_numbers', 'has_special_chars']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练机器学习模型（随机森林）
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测并评估模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("训练准确率 Accuracy:", accuracy)

# 检测新域名是否恶意
new_domain = "baidu.com"
print("测试域名: ", new_domain)
new_domain_features = pd.DataFrame({
    'length': [len(new_domain[0])],
    'has_numbers': [any(char.isdigit() for char in new_domain[0])],
    'has_special_chars': [any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",.<>?/" for char in new_domain[0])]
})
prediction = clf.predict(new_domain_features)
print("随机森林训练后预测结果 Prediction for new domain:", prediction)

# isblack=domain_label_map.__contains__(new_domain)
isblack=new_domain not in domain_label_map

print("黑名单查询结果:", isblack )