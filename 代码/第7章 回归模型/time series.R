library(forecast)
library(zoo)
library(DBI)
library(RPostgreSQL)

con = dbDriver("PostgreSQL")
connection<-dbConnect(con,host="你的主机名",user="你的用户名",password="你的密码",dbname="你的数据库名称",port="5432")

station= dbSendQuery(conn=connection,
                     statement="SELECT  *,(XKC+DKC+XHC+ZHC+DHC+TDH) AS volume
				 	                     FROM(SELECT GCRQ,(hour1::int)as HOUR,(minute1::int)as MINUTE,
								                            sum(XKC::int) as XKC, sum(DKC::int) as DKC, sum(XHC::int) as XHC, 
								                            sum(ZHC::int) as ZHC, sum(DHC::int) as DHC, sum(TDH::int) as TDH
						                       FROM s122
						                       WHERE XSFX='S'
						                       GROUP BY GCRQ,HOUR,MINUTE) a
          					            ORDER BY GCRQ,HOUR,MINUTE;") 
x<-strptime("2021-06-01 01:00:00","%Y-%m-%d %H:%M:%S")+300*1:NROW(station[,"volume"])
volume = zoo(station[,"volume"], x)
autoplot(volume)

par(mfrow=c(2,1))
acf(volume)
pacf(volume)

model.arima = auto.arima(volume_train, trace=TRUE, seasonal = TRUE,allowmean = FALSE)
summary(model.arima)
