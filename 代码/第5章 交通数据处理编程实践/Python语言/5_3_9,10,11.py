import pandas as pd 
raw_data = pd.read_csv('广东大坪站莞深一二期202106.csv')

# 将缺失值用0填充
raw_data_fillna = raw_data.fillna(0)


# 剔除有缺失值的行
raw_data_dropna = raw_data.dropna(axis = 0)
print('缺失数据共有', len(raw_data)-len(raw_data_dropna), '行')


# 检查原始数据重复情况，并存放检查结果 
judgement = raw_data_dropna.duplicated() 
# 创建一个空的DataFrame来存放重复数据，列名和原始数据相同
duplicated_data = pd.DataFrame(columns=raw_data_dropna.columns) 
for i in range(len(raw_data_dropna)):
    if judgement.iloc[i] == True: # 如果该行是重复数据
        # 将重复数据行加入到准备好的DataFrame，加入时忽略原数据的索引
        duplicated_data = duplicated_data.append(raw_data_dropna.iloc[i],ignore_index = True)
if len(duplicated_data) == 0:
    print("不存在完全一致的行")
else:
    print("有",len(duplicated_data),"行重复数据")
    new_data = raw_data_dropna.drop_duplicates() # 剔除重复行
    print("剔除重复数据后，有效数据有", len(new_data), "行")