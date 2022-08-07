# d.class <- read.csv("exercise.csv")
# d.class$sex = as.factor(d.class$sex)
# save(d.class, file = "d.class.Rdata")

load("d.class.Rdata")

# question (1)
new_data_1 <- subset(d.class, d.class$age >= 13)
rownames(new_data_1) <- 1:nrow(new_data_1)
print(new_data_1)

# question (2)
new_data_2 <- subset(d.class, (d.class$age >= 13 & d.class$sex == "F"))
new_data_2 <- new_data_2[,c(1,3)]
rownames(new_data_2) <- 1:nrow(new_data_2)
print(new_data_2)

# question (3)
x <- d.class$age
print(x)
