# -*- coding: utf-8 -*-
# 利用list构建一个矩阵，共五行，每一行第一列代表行驶里程，第二列代表行驶时间
count_vehicle=[[55,30],[45,25],[34,20],[60,40],[70,35]] # [[km,min],...]
speed=[] # 建立一个空list来存放前三辆车的速度
i=0
while i<=2: # 该辆车是否在前三辆
    speed_temp=count_vehicle[i][0]/(count_vehicle[i][1]/60)
    speed.append(int(speed_temp)) # 将计算出的速度放入list中
    i+=1 
average_speed=sum(speed)/len(speed) # 对前三辆车的速度求平均
print("前三辆车的平均速度是：",round(average_speed,2),"km/h")