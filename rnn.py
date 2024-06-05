import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
# 关闭警告
warnings.filterwarnings("ignore", category=UserWarning)
# 1. 收集域名数据
# 假设我们有一个包含域名和标签的数据集
data = pd.read_csv("domain_data_3w.csv")
data['domain'] = data['domain'].astype(str)

plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False   #The
# 2. 数据预处理
max_len = 50  # 假设最大域名长度为50个字符



# 使用字符级别的标记化和填充序列
char2idx = {char: idx+1 for idx, char in enumerate(set(''.join(data['domain'])))}
idx2char = {idx: char for char, idx in char2idx.items()}
X = [[char2idx[char] for char in domain] for domain in data['domain']]
X = pad_sequences(X, maxlen=max_len)

# 将标签编码为0和1
y = data['label'].values

# 存储训练数据量、准确率和召回率
train_sizes = []
train_accuracies = []
test_accuracies = []
train_recalls = []
test_recalls = []
# 3. 划分训练集和测试集
for train_size in range(5000, 80001, 5000):  # 逐步增加训练数据量
    X_train, X_test, y_train, y_test = train_test_split(X[:train_size], y[:train_size], test_size=0.2, random_state=42)

    # 4. 定义和训练模型
    model = Sequential()
    model.add(Embedding(len(char2idx)+1, 64))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

    # 5. 计算训练准确率
    train_accuracy = history.history['accuracy'][-1]

    # 6. 计算训练召回率
    train_preds = model.predict(X_train)
    train_preds = np.round(train_preds).astype(int)
    train_recall = recall_score(y_train, train_preds)

    # 7. 测试模型
    y_pred_probs = model.predict(X_test)
    y_pred = np.round(y_pred_probs).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    # 存储结果
    train_sizes.append(train_size)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    train_recalls.append(train_recall)
    test_recalls.append(test_recall)

print(train_sizes)
print(train_accuracies)
print(test_accuracies)
print(train_recalls)
print(test_recalls)

# 绘制准确率折线图
plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_accuracies, label='训练准确率')
plt.plot(train_sizes, test_accuracies, label='测试准确率')
plt.xlabel('训练数据量')
plt.ylabel('准确率')
plt.title('训练数据量与准确率的关系')
plt.legend()
# plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
# plt.show()

# 绘制召回率折线图
plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_recalls, label='训练召回率')
plt.plot(train_sizes, test_recalls, label='测试召回率')
plt.xlabel('训练数据量')
plt.ylabel('召回率')
plt.title('训练数据量与召回率的关系')
plt.legend()
# plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
plt.show()