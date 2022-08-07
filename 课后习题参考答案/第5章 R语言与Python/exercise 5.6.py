# 导入需要的包
import pandas as pd
# import psycopg2
from sqlalchemy import create_engine

# 导入数据
original_data = pd.read_csv('广东大坪站莞深一二期202106.csv')

########## 问题1 ##########
# 删除无用数据列
data = original_data.drop(['en_station_name','check_time','speed'],axis = 1)


########## 问题2 ##########
# 删除缺失值
data.dropna(inplace = True)
# 删除重复数据行
data = data.drop_duplicates()
# 重排索引
data = data.reset_index(drop = True)


########## 问题3 ##########
# 在dataframe中新建一列
data['flag'] = ''

for i in range(len(data)):
    if (data['weight'][i] > data['limit_weight'][i]):
        data['flag'][i] = 1
    else:
        data['flag'][i] = 0
        
data = data.drop(['limit_weight'],axis = 1)
   
number_of_overweight_vehicles = data['flag'].sum(axis = 0)
print("超重车辆有", number_of_overweight_vehicles, "辆")


########## 问题4 ##########
# 筛选超重及非超重数据
X_set = data.loc[data['flag'] == 0]
Y_data = data.loc[data['flag'] == 1]

# 从未超重数据集中随机选择与超重数据数据相同的数据并与超重数据集合并
X_data = X_set.sample(n = len(Y_data), replace = False, random_state = None, axis = 0)
new_data = X_data.append(Y_data)

# 创建连接对象
engine = create_engine('postgresql+psycopg2' + '://' 
                       + 'ACME_Lab' + ':' 
                       + '12345678' + '@' 
                       + 'localhost' + ':' 
                       + '5432' + '/' 
                       + 'Transportation Big Data')

new_data.to_sql(name = 'overweight_data', schema = 'public', con = engine, if_exists = 'replace', index = False)

read_data = pd.read_sql('SELECT * FROM overweight_data',con = engine)





