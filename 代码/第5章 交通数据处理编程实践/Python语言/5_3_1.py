# -*- coding: utf-8 -*-
distance=float(input('请输入行驶里程(单位：km)：')) #读入行驶里程
travel_time=float(input('请输入行驶时间（单位：min)：')) #读入行驶时间
speed=distance/(travel_time/60) #计算速度
print("汽车的速度是",speed,"km/h") #输出速度