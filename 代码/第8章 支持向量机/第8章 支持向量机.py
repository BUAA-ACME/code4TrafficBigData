# -*- coding: utf-8 -*-

# 导入需要的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import r2_score

# 从数据库导入数据  
engine = create_engine('postgresql+psycopg2'+'://' +
                       '你的登录名' +':' +
                       '你的登录密码' + "@" +
                       '你的主机名' + ':' +
                       '5432' + '/' +
                       '你的数据库名称'
                      )
data = pd.read_sql('SELECT * FROM real_time_data',con = engine) 

# 删除无用数据列
data = data.drop(['XKCS','YEAR','GCRQ','GCZBS','HOUR','MINUTE','SBSBM','DCSJLX',
                           'SJXH','CLZQ','DKCS','XHCS','ZHCS','DHCS','TDHS','JZXS','MTCS',
                           'TLJS','GCBFB','SJZYL','SJZYL','FLAGS','OFF_MINS','ERR_CODE',
                           'ERR_DESC','DELETE_BY','DELETE_TIME','CREATE_BY','CREATE_TIME',
                           'UPDATE_BY','UPDATE_TIME'],axis = 1)

# 删除缺失值
data.dropna(inplace = True)

# 删除重复数据行
data = data.drop_duplicates(keep = 'first')

# 重排索引
data = data.reset_index(drop = True)

# 新增总车流量列
data['flow'] = data['XKC'] + data['DKC'] + data['XHC'] + data['ZHC'] + data['DHC'] + data['TDH'] + data['JZX'] + data['MTC'] + data['TLJ']
                
# 筛选上行、11车道的数据
new_data = pd.DataFrame({'PJCTJJ':[],'flow':[]})
for index, row in data[:].iterrows():
    if data.at[index,'CDH'] == 11 and data.at[index,'XSFX'] == 'S':
        new_data.at[index,'PJCTJJ'] = data.at[index,'PJCTJJ']
        new_data.at[index,'flow'] = data.at[index,'flow']

# 重排索引
new_data = new_data.reset_index(drop = True)

# 设置自变量X和因变量y
X = new_data['PJCTJJ']
y = new_data['flow']

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

# 记录训练时间
t0 = time.time()

# 训练svr模型
svr.fit(X_train,y_train)

# 记录并输出训练时间
svr_fit = time.time() - t0
print(svr_fit)

# 模型预测
y_svr = svr.predict(X_test)

# 记录并输出预测时间
t0 = time.time()
svr_predict = time.time() - t0
print(svr_predict)

# 绘制训练集散点图
plt.scatter(X_train, y_train)
plt.ylabel('Flow',fontsize = 14)
plt.xlabel('PJCTJJ',fontsize = 14)
plt.legend()
plt.show()

# 绘制测试集真实值散点图
plt.scatter(X_test, y_test)
plt.ylabel('Flow',fontsize = 14)
plt.xlabel('PJCTJJ',fontsize = 14)
plt.legend()
plt.show()

# 绘制测试集预测值散点图
plt.scatter(X_test, y_svr)
plt.ylabel('Flow',fontsize = 14)
plt.xlabel('PJCTJJ',fontsize = 14)
plt.legend()
plt.show()

# 计算R2
print("得分:", r2_score(y_test, y_svr))