from sklearn.cluster import KMeans,AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

#从本地读取csv文件
data_or = pd.read_csv('g_205_6.csv')
print(data_or.head())
print(len(data_or))

#从数据库中获取所需数据
engine = create_engine('postgresql+psycopg2'+'://' +
                       '你的登录名' +':' +
                       '你的登录密码' + "@" +
                       '你的主机名' + ':' +
                       '5432' + '/' +
                       '你的数据库名称'
                      )
data_or = pd.read_sql('SELECT * FROM g205_6',con=engine)

#获取上层方向数据:S表示上层，X表示下层
data = data_or[data_or['XSFX']=='X'].reset_index()

#===========获取跟车百分比和平均车头间距数据=========
SJZYL_all = []  #用于存储最终时间占有率数据
PJCTJJ_all = [] #用于存储最终平均车头间距数据

for d in range(1,31):               #条件筛选：日期
    date_str = str(d)+'-Jun-21'
    for h in range(1,25):           #条件筛选：小时
        temp_list_speed = []
        for s in range(0,60,5):     #条件筛选：分钟时段
            #筛选满足条件的时间占有率数据
            SJZYL_temp = data[(data['GCRQ'] == date_str) &
                              (data['HOUR'] == h) &
                              (data['MINUTE'] == s)].loc[:,'SJZYL']
            # 筛选满足条件的平均车头间距数据
            PJCTJJ_temp =data[(data['GCRQ'] == date_str) &
                              (data['HOUR'] == h) &
                              (data['MINUTE'] == s)].loc[:, 'PJCTJJ']

            #判断是否有含0的数据
            if pd.isnull(np.mean(list(filter((0).__ne__, PJCTJJ_temp)))) or \
                    pd.isnull(np.mean(list(filter((0).__ne__, SJZYL_temp)))):
                continue
            else:
                PJCTJJ_all.append(np.mean(list(filter((0).__ne__, PJCTJJ_temp))))
                SJZYL_all.append(np.mean(list(filter((0).__ne__, SJZYL_temp))))

PJCTJJ = pd.Series(PJCTJJ_all,name='PJCTJJ')
SJZYL = pd.Series(SJZYL_all,name='SJZYL')
result = pd.concat([PJCTJJ,SJZYL],axis=1)  #将获取的数据组合成新的数据框

print(result)

#====依次创建不同聚类数目的模型并保存每个模型的SSE====
SSE_list = [ ]
K = range(1, 11)                    #设置拟聚类数目
for k in range(1,11):
    km_model=KMeans(n_clusters=k)   #分别建立不同聚类数目下的Kmeans模型
    km_model.fit(result)
    SSE_list.append(km_model.inertia_)   #将模型的误差平方和存入列表
    print(SSE_list)

#绘制聚类数目与SSE的关系曲线
plt.figure()
plt.plot(np.array(K), SSE_list, 'bx-')
plt.rcParams['figure.figsize'] = [12,8]
plt.xlabel('K',fontsize=15)
plt.ylabel('SSE',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#===============数据标准化===========
scaler = StandardScaler()
scaler.fit(result)
result_nor = scaler.transform(result)

#===========模型建立及标签获取=======
model = KMeans(n_clusters=3, random_state=1)    #建立K-means模型
#model = AgglomerativeClustering(n_clusters=3)  #建立层次聚类模型
y_pred = model.fit_predict(result_nor)

#==========原始数据及标签合并=========
cluster_output = pd.concat((result,     #愿式数据
                           pd.DataFrame(y_pred,columns=['labels'])),    #标签数据
                           axis=1       #按列合并
                           )

#==========最终聚类结果展示===========
plt.scatter(cluster_output[cluster_output['labels']==2]['PJCTJJ'],
            cluster_output[cluster_output['labels']==2]['SJZYL'],
            c='firebrick',label='拥堵')
plt.scatter(cluster_output[cluster_output['labels']==1]['PJCTJJ'],
            cluster_output[cluster_output['labels']==1]['SJZYL'],
            c='blue',label='轻微拥堵')
plt.scatter(cluster_output[cluster_output['labels']==0]['PJCTJJ'],
            cluster_output[cluster_output['labels']==0]['SJZYL'],
            c='orange',label='流畅')

plt.rcParams['figure.figsize'] = [12,8]
plt.xlabel('平均车头间距',fontsize=14)
plt.ylabel('时间占有率',fontsize=14)
plt.legend()
plt.show()