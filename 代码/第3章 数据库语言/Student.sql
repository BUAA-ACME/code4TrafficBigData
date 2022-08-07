-- 定义学生表Student
Create Table Student ( S# char(8) not null , Sname char(10),  
x char(2), Sage integer, D# char(2), Sclass char(6) ); 


-- 定义课程表 Course
Create Table Course ( C# char(3) , Cname char(12), Chours integer,  
dit float(1), T# char(3) );  

-- 向学生表中追加元组，追加两条记录
Insert Into Student  
Values ( ‘54601’ , ‘张三’, ‘男’, 20， ’03’， ‘546’);  
Insert Into Student ( S#, Sname, Ssex, Sage, D# , Sclass)  
Values ( ‘54602’ , ‘张四’, ‘女’, 20， ’03’， ‘546’); 


-- 将所有计算机系的教师工资上调10%
Update Teacher  
Set Salary = Salary * 1.1  
Where D# in  
( Select D# From Dept Where Dname = ‘计算机’); 




