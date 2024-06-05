import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False   #The
# 加载数据集
data = pd.read_csv("domain_data_3w.csv")
data['domain'] = data['domain'].astype(str)
blacklist = set()
whitelist = set()

# example:
# icb-online-intl.com,FALSE
# kobesportswear.date,FALSE
x_gap = 5000
for index, row in data.iterrows():
    if row['label'] == False:
        # domain_label_map[row['domain']] = row['']
        blacklist.add(row['domain'])
    else:
        whitelist.add(row['domain'])

print("白名单数量: ",len(whitelist))
print("黑名单数量: ",len(blacklist))


# 计算黑名单方法的准确率
def calculate_percentage(domain_set_a, domain_set_b):
    # Ensure sets for unique domains
    set_a = set(domain_set_a)
    set_b = set(domain_set_b)

    # Calculate the intersection of A and B
    intersection = set_a.intersection(set_b)

    # Calculate the percentage
    if len(set_a) == 0:
        return 0.0

    percentage = (len(intersection) / len(set_a))
    return percentage


# 提取特征
data['length'] = data['domain'].apply(len)
data['has_numbers'] = data['domain'].apply(lambda x: any(char.isdigit() for char in x))
data['has_special_chars'] = data['domain'].apply(lambda x: any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",<>?/" for char in x))

# 准备训练和测试数据，保留原始域名列
X = data[['domain', 'length', 'has_numbers', 'has_special_chars']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 从特征中移除域名列，用于模型训练和预测
X_train_features = X_train[['length', 'has_numbers', 'has_special_chars']]
X_test_features = X_test[['length', 'has_numbers', 'has_special_chars']]

# 定义和训练模型，或加载已保存的模型
model_filename = 'random_forest_model.pkl'
try:
    model = load(model_filename)
    print("模型加载成功")
except:
    print("无法加载模型，正在重新训练")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_features, y_train)
    dump(model, model_filename)
    print("模型训练完成并保存")

# 记录不同数据量下的训练准确率、测试准确率
train_accuracies = []
test_accuracies = []
blacklist_accuracies = []
data_sizes = []

# 分步增加训练数据量，每步10000个样本
for size in range(x_gap, 80001, x_gap):
    # 根据size确定训练数据量
    X_train_frac = X_train_features.iloc[:size]
    y_train_frac = y_train.iloc[:size]
    xt=     X_train.iloc[:size]
    # 重新训练模型
    model.fit(X_train_frac, y_train_frac)

    # 记录当前数据量
    data_sizes.append(len(X_train_frac))

    # 预测和记录准确率
    train_preds = model.predict(X_train_frac)
    test_preds = model.predict(X_test_features)

    tranpre = accuracy_score(y_train_frac, train_preds)
    testpre = accuracy_score(y_test, test_preds)
    train_accuracies.append(tranpre)
    test_accuracies.append(testpre)

    # 计算黑名单方法的准确率
    xx = set(xt.iloc[:, 0])
    bpre = calculate_percentage(xx, list(blacklist)[:size])
    blacklist_accuracies.append(bpre)
    print(f"样本数量 {size} 比例 {size/800:.4f}%: 训练准确率 {tranpre:.4f}.% 测试准确率 {testpre:.4f}%  黑名单准确率 {bpre:.4f}%")

# 绘制准确率折线图
plt.figure(figsize=(10, 5))

# 绘制训练和测试准确率折线图
plt.plot(data_sizes, train_accuracies, label='训练准确率')
plt.plot(data_sizes, test_accuracies, label='测试准确率')
plt.plot(data_sizes, blacklist_accuracies, label='黑名单准确率')
plt.xlabel('训练数据量')
plt.ylabel('准确率')
plt.title('黑名单和随机森林模型准确率对比')
plt.legend()
plt.gca().xaxis.set_major_locator(MultipleLocator(x_gap))  # 设置 X 轴间隔
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))  # 设置 Y 轴为百分比，间隔为 5%


plt.tight_layout()
plt.show()
