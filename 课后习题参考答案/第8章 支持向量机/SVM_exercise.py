# -*- coding: utf-8 -*-
# 第二题
from sklearn import svm
x=[[1, 2], [2, 5], [3, 3], [2, 1], [3, 2]]
y=[1, 1, 1, -1, -1]
clf = svm.SVC(kernel='linear',C=10000)
clf.fit(x, y)
print(clf.coef_)
print(clf.intercept_)

# 第三题
# 导入需要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# 导入数据
original_data = pd.read_csv('D:/交通大数据/教材编写/支持向量机/Speed_data.csv')
data = pd.DataFrame(original_data)

# 新增一列索引列
data['index'] = data.index

# 设置自变量X和因变量y
X = data['index']
y = data['Speed']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,train_size = 0.8)

# 转换数据类型
X_list = X_train.to_list()
X_array = np.array(X_list)
X_train = X_array.reshape(len(X_train),-1)

X_list = X_test.to_list()
X_array = np.array(X_list)
X_test = X_array.reshape(len(X_test),-1)


y_list = y_train.to_list()
y_array = np.array(y_list)
y_train = y_array.reshape(len(y_train),-1)

y_list = y_test.to_list()
y_array = np.array(y_list)
y_test = y_array.reshape(len(y_test),-1)

# 网格搜索最优参数
svr = GridSearchCV(SVR(kernel = 'rbf',gamma = 0.1), cv = 5, 
                   param_grid = {'C':[1e0, 1e1, 1e2, 1e3],'gamma':np.logspace(-2, 2, 5)})

# 训练svr模型
svr.fit(X_train,y_train)

# 模型预测
y_svr = svr.predict(X_test)

# 计算R2
print("得分:", r2_score(y_test, y_svr))

# 预测之后一小时的车速
X_predict = np.array([2976,2977,2978,2979])
X_predict = X_predict.reshape(-1, 1)
y_svr = svr.predict(X_predict)
print(y_svr)

# 第四题
# 导入需要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# 导入数据
original_data = pd.read_csv('D:/交通大数据/教材编写/支持向量机/Flow_data.csv')
data = pd.DataFrame(original_data)

# 新增一列索引列
data['index'] = data.index

# 设置自变量X和因变量y
X = data['index']
y = data['FLOW']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,train_size = 0.8)

# 转换数据类型
X_list = X_train.to_list()
X_array = np.array(X_list)
X_train = X_array.reshape(len(X_train),-1)

X_list = X_test.to_list()
X_array = np.array(X_list)
X_test = X_array.reshape(len(X_test),-1)


y_list = y_train.to_list()
y_array = np.array(y_list)
y_train = y_array.reshape(len(y_train),-1)

y_list = y_test.to_list()
y_array = np.array(y_list)
y_test = y_array.reshape(len(y_test),-1)

# 网格搜索最优参数
svr = GridSearchCV(SVR(kernel = 'rbf',gamma = 0.1), cv = 5, 
                   param_grid = {'C':[1e0, 1e1, 1e2, 1e3],'gamma':np.logspace(-2, 2, 5)})

# 训练svr模型
svr.fit(X_train,y_train)

# 模型预测
y_svr = svr.predict(X_test)

# 计算R2
print("得分:", r2_score(y_test, y_svr))

# 预测2014/4/1 8：00的车流量数
X_predict = np.array([3008])
X_predict = X_predict.reshape(-1, 1)
y_svr = svr.predict(X_predict)
print(y_svr)