# pip install -U pandas scikit-learn 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
from tkinter import ttk

# 读取数据集，数据集包含域名和标签（恶意或正常）
data = pd.read_csv("domain_data.csv")
data['domain'] = data['domain'].astype(str)

# 黑名单集合
# domain_label_map = {}
blacklist = set()

# example:
# icb-online-intl.com,FALSE
# kobesportswear.date,FALSE

for index, row in data.iterrows():
    if row['label'] == False:
        # domain_label_map[row['domain']] = row['label']
        blacklist.add(row['domain'])

print("黑名单数量: ",len(blacklist))

# 提取特征：例如，域名长度、是否包含数字、是否包含特殊字符等
data['length'] = data['domain'].apply(len)
data['has_numbers'] = data['domain'].apply(lambda x: any(char.isdigit() for char in x))
data['has_special_chars'] = data['domain'].apply(lambda x: any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",.<>?/" for char in x))

# 划分数据集为训练集和测试集
X = data[['length', 'has_numbers', 'has_special_chars']]
y = data['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt']
# }
# 训练机器学习模型（随机森林）
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf = RandomForestClassifier(n_estimators=100,max_features='sqrt',min_samples_leaf=4, min_samples_split=10)

# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')

# grid_search.fit(X_train, y_train)
# # 输出最佳参数
# print("Best Parameters:", grid_search.best_params_)

# # 输出最佳模型在验证集上的准确率
# print("Validation Accuracy:", grid_search.best_score_)
# exit(0)

clf.fit(X_train, y_train)

# 预测并评估模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("训练准确率 Accuracy:", accuracy)

# 检测新域名是否恶意
# new_domain = "zhongsou.net"
# print("测试域名: ", new_domain)
# new_domain_features = pd.DataFrame({
#     'length': [len(new_domain[0])],
#     'has_numbers': [any(char.isdigit() for char in new_domain[0])],
#     'has_special_chars': [any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",.<>?/" for char in new_domain[0])]
# })
# prediction = clf.predict(new_domain_features)
# print("随机森林训练后预测结果 Prediction for new domain:", prediction)

# isblack=domain_label_map.__contains__(new_domain)
# isblack=new_domain not in domain_label_map

# print("黑名单查询结果:", isblack )

# 创建一个示例的黑名单
# blacklist = {"malicious-domain.com", "example.com"}

# 创建主窗口
root = tk.Tk()
root.title("恶意域名检测")
root.geometry("800x600")  # 设置窗口大小为800x600


clf.fit(X_train, y_train)

# 定义函数：检测域名是否是恶意域名
def detect_malicious_domain():
    domain = entry.get()
    features = [[len(domain), any(char.isdigit() for char in domain), any(char in "!@#$%^&*()-_+=~`[]{}\\|;:'\",.<>?/" for char in domain)]]
    prediction = clf.predict(features)
    
    # 模型预测结果
    if prediction[0] == 1:
        model_result = "是恶意域名"
    else:
        model_result = "不是恶意域名"

    # 查询黑名单结果（这里用示例的方式表示，实际可以替换为真实的黑名单查询逻辑）
    blacklist_result = "未知"
    if domain in blacklist:
        blacklist_result = "在黑名单中"
    
    # 显示结果
    result_tree.insert("", tk.END, values=(domain, model_result, blacklist_result))

# 创建标签、输入框和按钮
label = tk.Label(root, text="请输入域名：")
label.pack()
entry = tk.Entry(root)
entry.pack()
button = tk.Button(root, text="检测", command=detect_malicious_domain)
button.pack()

# 创建Treeview用于显示结果
columns = ("Domain", "Model Prediction", "Blacklist Result")
result_tree = ttk.Treeview(root, columns=columns, show="headings")
result_tree.heading("Domain", text="域名")
result_tree.heading("Model Prediction", text="模型预测结果")
result_tree.heading("Blacklist Result", text="黑名单查询结果")
result_tree.pack()


# 运行主事件循环
root.mainloop()