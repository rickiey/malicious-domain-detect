import pandas as pd
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import warnings
# 关闭警告
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False   #The
# 2. 数据预处理
# 存储训练数据量、准确率和召回率
train_sizes =[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
train_accuracies = [0.8370000123977661, 0.8529999852180481, 0.887583315372467, 0.8505625128746033, 0.8791000247001648, 0.8989583253860474, 0.9140357375144958, 0.9254999756813049, 0.9341111183166504, 0.9402499794960022, 0.8655454516410828, 0.7985000014305115, 0.7439423203468323, 0.7032142877578735, 0.6791666746139526, 0.6763281226158142]
test_accuracies =[0.884, 0.891, 0.9166666666666666, 0.86325, 0.8822, 0.9056666666666666, 0.9148571428571428, 0.921, 0.9312222222222222, 0.9394, 0.8643636363636363, 0.798, 0.7512307692307693, 0.7165, 0.6980666666666666, 0.6988125]
train_recalls = [0.9502716255745925, 0.8377248013383521, 0.756779307467668, 0.18881987577639753, 0.019206680584551147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006779661016949152, 0.08855902251842876, 0.20841437936748072, 0.28227558923496965, 0.7201912132938766, 0.841761638014294]
test_recalls = [0.949748743718593, 0.8313856427378965, 0.7706576728499157, 0.1791304347826087, 0.020168067226890758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0066711140760507, 0.07211538461538461, 0.20778851620238772, 0.2742587601078167, 0.7096061832903938, 0.8365059871046976]

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
plt.title('RNN 训练数据量与准确率的关系')
plt.legend()
# plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
# plt.show()

# 绘制召回率折线图
plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_recalls, label='训练召回率')
plt.plot(train_sizes, test_recalls, label='测试召回率')
plt.xlabel('训练数据量')
plt.ylabel('召回率')
plt.title('RNN 训练数据量与召回率的关系')
plt.legend()
# plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
plt.show()