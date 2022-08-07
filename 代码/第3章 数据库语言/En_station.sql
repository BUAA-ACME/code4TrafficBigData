-- 查询检查时间为“19：01：50”在检查站的检查时间，车辆类型，载重，总轴数等相关属性。
SELECT Check_time, Vehicle_type, Weight, Alex_count 
FROM En_station
WHERE Check_time = “19:01:50 ”; 


-- 对检查时间为“19:01:50 ”的所有属性进行查询
SELECT *  
FROM En_station  
WHERE Check_time = “19:01:50”; 


-- 对结果中的列进行重命名
SELECT Check_time AS Time, Vehicle_type AS type, Weight  
FROM En_station  
WHERE title = “19:01:50 ”; 


-- LIKE的语法规则
SELECT Check_time, Vehicle_type, Weight, Alex_count   
FROM En_station  
WHERE Check_time LIKE '19:01__'  


-- 选择载重大于8000并且车轴数量大于2的记录，或者限重高于18000的记录
SELECT Check_time, Vehicle_type, Weight, Alex_count, Limit_weight  
FROM En_station 
WHERE Limit_weight > 18000 OR Alex_count > 2 AND Weight > 8000 


-- 查找载重大于8000的记录，并按其分级对车辆进行排序。
SELECT Check_time, Vehicle_type, Weight, Limit_weight
FROM En_station 
WHERE Weight > 8000
ORDER BY Vehicle_type;  


-- 根据车辆类型进行正序排列，如果车辆类型相同，那么将车辆类型相同的两条记录进行载重的倒序排列。
SELECT Check_time, Vehicle_type, Weight, Limit_weight  
FROM En_station
WHERE Weight > 8000  
ORDER BY Vehicle_type ASC, Weight DESC  


--去除重复值
SELECT DISTINCT Vehicle_type 
FROM En_station



