select s1 as project_id ,s2 as employee_id  
from(  
select s1,s2,  
dense_rank() over(partition by s1  
order by s3 desc ) as r  
from  
(select a.project_id as s1,a.employee_id as s2,  
b.experience_years as s3  
from project a  
join employee b  
on a.employee_id=b.employee_id)c  
)d  
where r=1; 
