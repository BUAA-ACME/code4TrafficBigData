# -*- coding: utf-8 -*-
# 第一题
# 导入包
import numpy as np
import matplotlib.pyplot as plt

# 节点定义
class TreeNode(object):
    def __init__(self, tempR, tempc):
        self.R = tempR
        self.c = tempc
        self.left = None
        self.right = None

# 输入y
y = np.array([4.5, 4.78, 4.91, 5.32, 5.8, 7.05, 7.8, 8.23, 8.72, 9])

# 利用CART算法建立回归树
# 变换切分点n，选择使得平方误差最小的切分点
def CART(start, end):
    # 切点n的选择表示R1为x值小于等于n的点，R2为大于n的点
    if(end - start >= 1):
        result = []
        for n in range(start+1, end+1): # n在(start, end]之间取值
            y1 = y[start:n]  # y1取索引为[start, n]之间的值
            y2 = y[n:end+1]  # y2取索引为[n+1, end]之间的值
            result.append((y1.std()**2)*y1.size + (y2.std()**2)*y2.size)
            # std即标准差函数，求标准差的时候默认除以元素的个数
            # 因此平方后乘以元素个数才是要求的平方差
        index1 = result.index(min(result)) + start # 取平方差误差最小的索引值
        root = TreeNode(y[start:end+1], min(result))
        # 索引值为0-9，x值为1-10，即n的值比求的索引值多1
        print("节点元素值为",y[start:end+1], "  n =",index1+1, 
              "  最小平方误差为",min(result)) # 输出n值和最小平方误差
        root.left = CART(start, index1)  # 对列表的左侧生成左子树
        root.right = CART(index1+1, end)  # 对列表的右侧生成右子树
    else:
        root = None
    return root


if __name__ == "__main__":
    root = CART(0, 9)
    
# 第二题
# 导入包
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
import pandas as pd

# 定义转换函数
def weather_type(s):
    it = {'sunny':1, 'rainy':2, 'foggy':3, 'snowy':4}
    return it[s]
def time(s):
    it = {'peak_hour':1, 'non_peak_hour':2}
    return it[s]
def holiday(s):
    it = {'yes':1, 'no':0}
    return it[s]
def road_quality(s):
    it = {'good':1, 'bad':0}
    return it[s]
def special_weather(s):
    it = {'yes': 1, 'no': 0}
    return it[s]
def traffic_condition(s):
    it = {'good': 1, 'bad': 0}
    return it[s]

traffic_feature_E = 'weather_type', 'time', 'holiday','road_quality', 'special_weather'
traffic_class = 'good', 'bad'

# 导入数据
data = pd.read_csv("C:/Users/11/Desktop/DT/traffic_condition.csv")

# 将原始数据中的数据转换为数字形式
for i in range(8):
    data.iloc[i]['weather_type'] = weather_type(data.iloc[i]['weather_type'])
    data.iloc[i]['time'] = time(data.iloc[i]['time'])
    data.iloc[i]['holiday'] = holiday(data.iloc[i]['holiday'])
    data.iloc[i]['road_quality'] = road_quality(data.iloc[i]['road_quality'])
    data.iloc[i]['special_weather'] = special_weather(data.iloc[i]['special_weather'])
    data.iloc[i]['traffic_condition'] = traffic_condition(data.iloc[i]['traffic_condition'])
    
# 划分x、y
y = data['traffic_condition']
x = data.drop(['traffic_condition'],axis = 1)

# 拆分训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 使用信息熵作为划分标准，对决策树进行训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train.astype('int'))

# 把决策树结构写入文件
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=traffic_feature_E, class_names=traffic_class,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('play1.pdf')

# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
print(clf.feature_importances_)

# 转换数据类型
y_list = y_train.to_list()
y_array = np.array(y_list)
y_train = y_array.reshape(len(y_train),-1)

y_list = y_test.to_list()
y_array = np.array(y_list)
y_test = y_array.reshape(len(y_test),-1)

# 使用训练数据预测
answer = clf.predict(x_train)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

# 对测试数据进行预测
answer = clf.predict(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))

# 第三题
# 导入包
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 导入数据
original_data = pd.read_csv('C:/Users/11/Desktop/DT/Speed_data.csv')
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

# 模型训练
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# 模型预测
regressor_predict = regressor.predict(X_test)

#  模型评估
print("回归树的R_squared值为：", r2_score(y_test, regressor_predict))

# 预测之后半小时的车速
X_predict = np.array([2976,2977])
X_predict = X_predict.reshape(-1, 1)
y_dt = regressor.predict(X_predict)
print(y_dt)

# 第四题
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

# 模型训练
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# 模型预测
regressor_predict = regressor.predict(X_test)

#  模型评估
print("回归树的R_squared值为：", r2_score(y_test, regressor_predict))

# 预测2014年4月2日1:00的统计交通流量
X_predict = np.array([2980])
X_predict = X_predict.reshape(-1, 1)
y_dt = regressor.predict(X_predict)
print(y_dt)