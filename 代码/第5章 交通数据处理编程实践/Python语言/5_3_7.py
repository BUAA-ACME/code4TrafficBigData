# -*- coding: utf-8 -*-
# 构建计算车辆行驶速度的函数：get_speed
def get_speed (count_data):
    cal_speed=[] # 建立一个空list来存放计算后的车辆速度
    for i in range(len(count_data)): # 遍历矩阵中的每一行
        speed_temp=count_data[i][0]/(count_data[i][1]/60)
        cal_speed.append(int(speed_temp)) # 将计算出的速度放入list中
    return cal_speed # 返回计算得到的各个车辆行驶速度
# 主函数
if __name__=='__main__':
    # 利用list构建一个矩阵，共五行，每一行第一列代表行驶里程，第二列代表行驶时间
    count_vehicle=[[55,30],[45,25],[34,20],[60,40],[70,35]] # [[km,min],…]
    speed=get_speed(count_vehicle) # 调用get_speed函数，得到车辆行驶速度
    print(len(count_vehicle),"辆车的速度分别是：",speed)