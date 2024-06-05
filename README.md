
# 安装

## 安装 python 后 安装需要的包

> 如果pip下载慢，更新并配置清华镜像源

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

执行命令  `pip install -U pandas scikit-learn` 安装这两个包

> 或临时使用清华源安装
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pandas scikit-learn matplotlib

训练数据damain_data.csv 就是两列，domain就是黑名单和白名单域名，label 就是黑名单域名对应 FALSE ，表示危险域名，白名单的就是TRUE，表示正常域名

运行：

- Windows

> python main.py

- ~Unix

> python3 main.py
