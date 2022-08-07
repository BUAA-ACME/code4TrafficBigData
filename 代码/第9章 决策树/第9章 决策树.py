# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:07:11 2021

@author: 11
"""

# 导入需要的包
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import psycopg2
import graphviz 

# 从数据库导入数据
engine = create_engine('postgresql+psycopg2'+'://' +
                       '你的登录名' +':' +
                       '你的登录密码' + "@" +
                       '你的主机名' + ':' +
                       '5432' + '/' +
                       '你的数据库名称'
                      )
data = pd.read_sql('SELECT * FROM limit_data',con = engine)

# 设置自变量和因变量
y = data['flag']
X = data.drop(['flag'],axis = 1)

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
dot_data = tree.export_graphviz(clf, 
                                out_file = None, 
                                feature_names = ['vehicle_type','alex_count','weight'],
                                class_names = ['1','0'],
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph.render(view=True, format="pdf", filename="decisiontree_pdf")