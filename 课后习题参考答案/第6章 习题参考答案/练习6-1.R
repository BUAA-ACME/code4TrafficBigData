lambda = 8 
# 在R中，pois表示泊松分布，加上不同的前缀表示不同的函数，
# 加上前缀d表示概率密度函数
# dpois(x,lambda) 发生x次随机事件的概率
dpois(10,lambda)+dpois(9,lambda)+dpois(8,lambda)+dpois(7,lambda)
