# -*- coding: utf-8 -*-
# 利用list构建一个矩阵，共五行，每一行第一列代表行驶里程，第二列代表行驶时间
count_vehicle=[[55,30],[45,25],[34,20],[60,40],[70,35]] # [[km,min],...]
speed=[] # 建立一个空list来存放五辆车的速度
for i in range(len(count_vehicle)): # 遍历矩阵中的每一行
    speed_temp=count_vehicle[i][0]/(count_vehicle[i][1]/60)
    speed.append(int(speed_temp)) # 将计算出的速度放入list中
print("五辆车的速度分别是：",speed)