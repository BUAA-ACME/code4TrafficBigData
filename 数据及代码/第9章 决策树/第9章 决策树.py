# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:07:11 2021

@author: 11
"""

# 导入需要的包
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 导入数据
original_data = pd.read_csv('C:/Users/11/Desktop/教材修改/广东大坪站莞深一二期202106.csv')
original_data = pd.DataFrame(original_data)

# 删除无用数据列
data = original_data.drop(['en_station_name','check_time','speed'],axis = 1)

# 删除缺失值
data.dropna(inplace = True)

# 删除重复数据行
data = data.drop_duplicates()

# 重排索引
data = data.reset_index(drop = True)

# 新增标志列
for index,row in data[:].iterrows():
    if data.at[index,'weight'] > data.at[index,'limit_weight']:
        data.at[index,'flag'] = 1
    else:
        data.at[index,'flag'] = 0
    
# 检查超重数据量
count = 0
for i in range(25147):
    if data['flag'][i] == 1:
        count += 1
print(count)

# 筛选超重及非超重数据
X_set = data.loc[data['flag'] == 0]
Y_data = data.loc[data['flag'] == 1]

# 从未超重数据集中随机选择398条数据并与超重数据集合并
X_data = X_set.sample(n = 398, replace = False, random_state = None, axis = 0)
new_data = X_data.append(Y_data)

y = new_data['flag']
X = new_data.drop(['weight','flag','limit_weight'],axis = 1)

# 数据类型转化 
y_list = y.to_list()
y_array = np.array(y_list)
y = y_array.reshape(len(y),-1)
X = X.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,train_size = 0.8)

# 训练模型
clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 4)
print(clf)
clf.fit(X_train,y_train)

# 训练集上进行模型预测及评价正确率
answer = clf.predict(X_train)
y_train = y_train.reshape(-1)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

# 测试集上进行模型预测及评价正确率
answer = clf.predict(X_test)
y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print(np.mean(answer == y_test))

# 导出决策树
import graphviz 

dot_data = tree.export_graphviz(clf, 
                                out_file = None, 
                                feature_names = ['vehicle_type','alex_count'],
                                class_names = ['1','0'],
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render(view=True, format="pdf", filename="decisiontree_pdf")