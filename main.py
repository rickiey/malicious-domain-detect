# pip install -U pandas scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score
# from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.ticker import MultipleLocator

# import sklearn.external.joblib as extjoblib
import joblib

# 读取数据集，数据集包含域名和标签（恶意或正常）
data = pd.read_csv("domain_data_3w.csv")
data['domain'] = data['domain'].astype(str)

plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False   #The
# 黑名单集合
# domain_label_map = {}
blacklist = set()
whitelist = set()

# example:
# icb-online-intl.com,FALSE
# kobesportswear.date,FALSE
x_gap = 5000
for index, row in data.iterrows():
    if row['label'] == False:
        # domain_label_map[row['domain']] = row['label']
        blacklist.add(row['domain'])
    else:
        whitelist.add(row['domain'])

print("白名单数量: ",len(whitelist))
print("黑名单数量: ",len(blacklist))

# 提取特征：例如，域名长度、是否包含数字、是否包含特殊字符等
data['length'] = data['domain'].apply(len)
data['has_numbers'] = data['domain'].apply(lambda x: any(char.isdigit() for char in x))
data['has_special_chars'] = data['domain'].apply(lambda x: any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",.<>?/" for char in x))

# 划分数据集为训练集和测试集
X = data[['length', 'has_numbers', 'has_special_chars']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义和训练模型，或加载已保存的模型
model_filename = 'random_forest_model.pkl'
try:
    model = joblib.load(model_filename)
    print("模型加载成功")
except:
    print("无法加载模型，正在重新训练")
    # model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model = RandomForestClassifier(n_estimators=100,max_features='sqrt',min_samples_leaf=4, min_samples_split=10)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    print("模型训练完成并保存")

# 记录不同数据量下的训练准确率、测试准确率、训练召回率和测试召回率
train_accuracies = []
test_accuracies = []
train_recalls = []
test_recalls = []
data_sizes = []

# 分步增加训练数据量，每步5000个样本
for size in range(x_gap, 80001, x_gap):
    # 根据size确定训练数据量
    X_train_frac = X_train[:size]
    y_train_frac = y_train[:size]

    # 重新训练模型
    model.fit(X_train_frac, y_train_frac)

    # 记录当前数据量
    data_sizes.append(len(X_train_frac))

    # 预测和记录准确率、召回率
    train_preds = model.predict(X_train_frac)
    test_preds = model.predict(X_test)
    accuracy_s = accuracy_score(y_train_frac, train_preds)
    recall_sc = recall_score(y_train_frac, train_preds)
    print(f"{size} 训练准确率  {accuracy_s:.4f}   训练召回率 {recall_sc:.4f}" )

    train_accuracies.append(accuracy_score(y_train_frac, train_preds))
    test_accuracies.append(accuracy_score(y_test, test_preds))
    train_recalls.append(recall_score(y_train_frac, train_preds))
    test_recalls.append(recall_score(y_test, test_preds))

# 绘制准确率和召回率折线图
plt.figure(figsize=(18, 10))
plt.subplot(1, 2, 1)

plt.plot(data_sizes, train_accuracies, label='训练准确率')
plt.plot(data_sizes, test_accuracies, label='测试准确率')
plt.xlabel('训练数据量')
plt.ylabel('准确率')
plt.title('训练和测试准确率随数据量的变化')
plt.legend()
plt.gca().xaxis.set_major_locator(MultipleLocator(x_gap))  # 设置 X 轴间隔
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))  # 设置 Y 轴间隔
plt.subplot(1, 2, 2)
plt.plot(data_sizes, train_recalls, label='训练召回率')
plt.plot(data_sizes, test_recalls, label='测试召回率')
plt.xlabel('训练数据量')
plt.ylabel('召回率')
plt.title('训练和测试召回率随数据量的变化')
plt.legend()

plt.gca().xaxis.set_major_locator(MultipleLocator(x_gap))  # 设置 X 轴间隔
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))  # 设置 Y 轴间隔

plt.tight_layout()
plt.show()