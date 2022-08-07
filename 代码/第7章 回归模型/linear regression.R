library(DBI)
library(RPostgreSQL)

con = dbDriver("PostgreSQL")
connection<-dbConnect(con,host="你的主机名",user="你的用户名",password="你的密码",dbname="你的数据库名称",port="5432")

station= dbSendQuery(conn=connection, 
                     statement="SELECT*FROM(SELECT GCRQ,hour1::int as HOUR,minute1::int as MINUTE,
				  XKC::int as XKC,DKC::int as DKC,XHC::int as XHC,
				  ZHC::int as ZHC,DHC::int as DHC,TDH::int as TDH,
				  ((XKC::int)+(DKC::int)+(XHC::int)+(ZHC::int)+(DHC::int)+(TDH::int)) as volume,
				  PJCTJJ::int as headway,SJZYL::int as time_ocp FROM s122) a
				  WHERE volume>0 AND headway>0 AND time_ocp>0
				  ORDER BY GCRQ,HOUR,MINUTE;")


model.linear = lm(log(volume)~log(PJCTJJ)+log(SJZYL), data = station)  
summary(model.linear)
