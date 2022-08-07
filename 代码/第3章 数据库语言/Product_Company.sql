--表连接
SELECT pname, price  
FROM product, company  
WHERE manufacturer = cname AND country = 'Japan'  


--where连接操作
SELECT *  
FROM product, company  
WHERE manufacturer = cname 


-- JOIN或者INNER JOIN连接
SELECT *  
FROM product JOIN company  
ON manufacturer = cname 


--选出学生所在学校的地址
SELECT DISTINCT student.name, university.address  
FROM student, university  
WHERE student.studyat = university.name 


--找出所有生产“Gadgets”的公司名称和股票价格
SELECT DISTINCT c.cname, stockprice  
FROM (SELECT cname, stockprice  
       FROM company, product  
      WHERE cname = manufacturer   
            AND product.category = 'gadgets'  
    ) AS c 


--查找比名为“GizmoWorks”的公司生产的所有产品都贵的产品。
SELECT pname  
FROM product  
WHERE price > ALL (SELECT price  
                     FROM product  
                    WHERE manufacturer = 'GizmoWorks') 



-- 分别选出制造商为GizmoWorks的产品的平均价格以及GizmoWorks所生产产品的数量。
SELECT AVG(price)  
FROM product  
WHERE manufacturer = 'GizmoWorks'  


--对取平均后的结果进行重命名
SELECT COUNT(*) AS ProductCount  
FROM product  
WHERE manufacturer = 'GizmoWorks'  


----------------------------------------------------------------------------------
--外键约束
--主键声明
CREATE TABLE Product(  
Pname    CHAR(30) PRIMARY KEY,   
Category CHAR(30),  
Price    FLOAT,  
CName    CHAR(30) REFERENCES Company(CName) 


--外键声明
CREATE TABLE Product(  
Pname    CHAR(30) PRIMARY KEY,   
Category CHAR(30),  
Price    FLOAT,  
CName    CHAR(30),   
FOREIGN KEY (Cname) REFERENCES Company(CName)  
)  

--更新语句
CREATE TABLE Product(  
 PName    CHAR(30) PRIMARY KEY,   
 Category CHAR(30),  
 Price    FLOAT,  
 CName    CHAR(30),  
 FOREIGN KEY (CName) REFERENCES Company(CName)  
 ON DELETE SET NULL  
 ON UPDATE CASCADE  
)  







