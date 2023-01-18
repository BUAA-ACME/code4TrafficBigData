import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier   #用于建立集成学习模型
from sklearn.model_selection import train_test_split                #用于将将分割数据集为训练集和测试集
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay #混淆矩阵数据获取和展示
import matplotlib.pyplot as plt                                     #可视化展示
from matplotlib.pyplot import MultipleLocator                       #设置轴刻度间隔

#从本地读取数据
data = pd.read_csv('limit_data.csv')

#从数据库中获取所需数据
engine = create_engine('postgresql+psycopg2'+'://' +
                      'ACME_Lab' +':' +
                      '********' + "@" +
                      'localhost' + ':' +
                      '5432' + '/' +
                      'Transportation Big Data'

)
data_or = pd.read_sql('SELECT * FROM limit_data',con=engine)

X = data.loc[:,['vehicle_type','weight','alex_count','limit_weight']]   #特征变量选取
Y = data.loc[:,['flag']]                                                #标签变量选取
#特征变量分割为训练集和测试集
X_train, X_test= train_test_split(X,                #特征变量数据集
                                  random_state=1,   #获得可复制的结果
                                  test_size=0.25    #测试集占比
                                  )
#标签变量分割为训练集和测试集
y_train, y_test = train_test_split(Y,               #标签变量数据集
                                   random_state=1,  #获得可复制的结果
                                   test_size=0.25   #测试集占比
                                   )

train_result = []   #存储训练集准确率
test_result = []    #存储测试集准确率
for i in range(1,11):
    clf = AdaBoostClassifier(n_estimators=i,  #设定子学习器数目
                            random_state=1    #随机种子，用于获取可复制的结果
                             )
    # clf = BaggingClassifier(n_estimators=i, #设定子学习器数目
    #                         n_jobs=-1,      #使用所有处理器训练
    #                         random_state=1  #随机种子，用于获取可复制的结果
    #                         )
    clf.fit(X_train, y_train)                 #模型训练
    train_score = clf.score(X_train, y_train) #训练集准确率
    test_score = clf.score(X_test, y_test)    #测试集准确率
    train_result.append(train_score)
    test_result.append(test_score)

print(train_result)
print(test_result)

print(clf.score(X_test, y_test))
y_pre_train = clf.predict(X_train)
y_train = np.array(y_train)
y_train = y_train.reshape(-1)
print(np.mean(y_pre_train == y_train))
print(confusion_matrix(y_train, y_pre_train))

#=======不同数量基学习器下的模型性能展示============
def plot_train_test_score_curve(train_result,test_result):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))        #设置绘图画布大小
    x_value = list(range(1,11))                         #X轴刻度设置
    plt.plot(x_value,train_result)                      #绘制训练集准确率
    plt.plot(x_value,test_result)                       #绘制测试集准确率
    plt.legend(["Train Result","Test Result"])          #设置legend
    fontdict = {'family':'Microsoft YaHei','size':12}   #坐标轴字体格式设置
    plt.xlabel('nEstimator',fontdict=fontdict)          #X坐标轴标签设置
    plt.ylabel('Score',fontdict=fontdict)               #Y坐标轴标签设置
    x_major_locator=MultipleLocator(1)                  #X轴刻度间隔设置
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator=MultipleLocator(0.01)               #Y轴刻度间隔设置
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()                                          #图像显示

#plot_train_test_score_curve(train_result,test_result)

# clf = AdaBoostClassifier(n_estimators=10,  # 设定子学习器数目
#                          random_state=1  # 随机种子，用于获取可复制的结果
#                          ).fit(X_train, y_train)
clf = BaggingClassifier(n_estimators=10,   #设定子学习器数目
                        n_jobs=-1,        #
                        random_state=1    #随机种子，用于获取可复制的结果
                        ).fit(X_train, y_train)
# ============混淆矩阵图展示===========
y_pre = clf.predict(X_test)             #测试集分类预测
C2 = confusion_matrix(y_test, y_pre)    #构建混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=C2,          #指定混淆矩阵数据
                              display_labels=clf.classes_   #指定类标签
                              )
disp.plot()     #图像绘制
plt.show()      #图像展示

#print(balanced_accuracy_score(y_test, y_pre))