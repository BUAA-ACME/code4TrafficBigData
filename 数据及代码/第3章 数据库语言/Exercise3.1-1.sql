select name  
from member,book,borrowed   
where member.num=borrowed.num  
and book.isbn=borrowed.isbn   
and borrowed.publisher="人民交通出版社" 
