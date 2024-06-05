import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer

# 1. 收集域名数据
# 假设我们有一个包含域名和标签的数据集
data = pd.read_csv("domain_data_3w.csv")
data['domain'] = data['domain'].astype(str)

# 2. 提取域名特征
# 这里我们使用域名的长度作为特征，你也可以使用其他特征
data['length'] = data['domain'].apply(len)

# 将域名转换为向量，使用字符计数向量化器
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # 使用字符级别的3-gram特征
X = vectorizer.fit_transform(data['domain'])

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)

# 4. 训练模型
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 5. 测试模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("模型准确率：", accuracy)
print("模型召回率：", recall)

# 绘制准确率和召回率折线图
thresholds = np.linspace(0, 1, 10)  # 设定不同的阈值范围
precisions = []
recalls = []

for threshold in thresholds:
    y_pred_threshold = (model.predict_proba(X_test)[:,1] > threshold).astype(int)
    precision = accuracy_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    precisions.append(precision)
    recalls.append(recall)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='准确率')
plt.plot(thresholds, recalls, label='召回率')
plt.xlabel('阈值')
plt.ylabel('评分')
plt.title('准确率和召回率随阈值变化')
plt.legend()
plt.show()
