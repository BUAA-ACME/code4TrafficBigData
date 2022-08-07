select name,publisher   
from member,borrowed,book   
group by name,publisher   
where member.num=borrowed.num  
and  
book.isbn=borrowed.isbn  
having  
count(*)>=5  
