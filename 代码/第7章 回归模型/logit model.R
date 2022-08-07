library(DBI)
library(RPostgreSQL)

con = dbDriver("PostgreSQL")
connection<-dbConnect(con, host="你的主机名",user="你的用户名",password="你的密码",dbname="你的数据库名称",port = "5432")
acc_data = dbReadTable(conn=connection,name="A4_AccidentCount",value=data)

acc_data$is_acc = acc_data$AccCount>0
summary(acc_data)

model.logit = glm(is_acc~ST_MP+Length+NLane+LaneWidth+LShoulderWidth+RShoulderWidth+AADT, 		data=acc_data, family=binomial())
summary(model.logit)

model.new = glm(is_acc~ST_MP+Length+NLane+AADT,data=acc_data, family=binomial())
summary(model.new)
