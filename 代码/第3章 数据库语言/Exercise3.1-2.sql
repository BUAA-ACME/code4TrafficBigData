select name   
from member where num  
in  
(select num   
from borrowed   
group by num   
having count(*) =   
(select count(*)   
from book   
where publisher="人民交通出版社"))  
