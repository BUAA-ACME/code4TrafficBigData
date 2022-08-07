--找到每个产品的总销售额
SELECT Product, SUM(price * quantity) AS TotalSale  
FROM sale  
GROUP BY Product  

--嵌套子查询
SELECT DISTINCT x.Product,  
     (SELECT SUM(price * quantity) FROM sale AS y  
       WHERE x.product = y.product) AS TotalSale  
FROM sale AS x  


--查找2016年10月1日后售出且总销售量超过30个的每个产品的产品名称和总销售额。
SELECT Product, SUM(price * quantity) AS TotalSale  
FROM sale  
WHERE date > '2016-10-1'  
GROUP BY Product  
HAVING SUM(quantity) > 30  



