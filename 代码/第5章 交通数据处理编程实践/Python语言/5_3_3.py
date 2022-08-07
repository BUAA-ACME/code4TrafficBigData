# -*- coding: utf-8 -*-
distance=float(input('请输入行驶里程(单位：km)：')) #读入行驶里程
travel_time=float(input('请输入行驶时间（单位：min)：')) #读入行驶时间
speed=distance/(travel_time/60) #计算速度
if speed>120:
    print("您的当前车速为:",int(speed),"km/h，您已超速")
else:
    print("您的当前车速为:",int(speed),"km/h，请保持车速")