library(forecast)
library(zoo)
library(DBI)
library(RPostgreSQL)

con = dbDriver("PostgreSQL")
connection<-dbConnect(con,host="111.206.188.9",user="ACME_Lab",password="***",dbname="Transporation Big Data",port="5432")

station= dbSendQuery(conn=connection,
                     statement="SELECT  *,(XKC+DKC+XHC+ZHC+DHC+TDH) AS volume
				 	                     FROM(SELECT GCRQ,(hour1::int)as HOUR,(minute1::int)as MINUTE,
								                            sum(XKC::int) as XKC, sum(DKC::int) as DKC, sum(XHC::int) as XHC, 
								                            sum(ZHC::int) as ZHC, sum(DHC::int) as DHC, sum(TDH::int) as TDH
						                       FROM s122
						                       WHERE XSFX='S¡¯
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
