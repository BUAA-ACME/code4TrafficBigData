data <- read.csv("input.csv")
install.packages("RPostgreSQL")
library(DBI)
library(RPostgreSQL)
connection <- dbConnect(PostgreSQL(), host="你的主机名", user= "你的登录名", password="你的登录密码", dbname="你的数据库名称",port = "5432") 
dbWriteTable(conn = connection, name = "real_time_data", value = data)
read_data = dbReadTable(connection,"real_time_data")