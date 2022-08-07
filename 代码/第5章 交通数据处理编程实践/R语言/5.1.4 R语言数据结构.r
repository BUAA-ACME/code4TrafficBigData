# Create two 2x3 matrices.
matrix1 <- matrix(c(3, 9, -1, 4, 2, 6), nrow = 2)
print(matrix1)
matrix2 <- matrix(c(5, 2, 0, 9, 3, 4), nrow = 2)
print(matrix2)


# Add the matrices.
result <- matrix1 + matrix2
cat("Result of addition")
print(result)
# Subtract the matrices
result <- matrix1 - matrix2
cat("Result of subtraction")
print(result)
# Multiply the matrices.
result <- matrix1 * matrix2
cat("Result of multiplication")
print(result)
# Divide the matrices
result <- matrix1 / matrix2
cat("Result of division")
print(result)


vector1 <- c(5,9,3)
vector2 <- c(10,11,12,13,14,15)
# Take these vectors as input to the array.
array1 <- array(c(vector1,vector2),dim = c(3,3,2))
# Create two vectors of different lengths.
vector3 <- c(9,1,0)
vector4 <- c(6,0,11,3,14,1,2,6,9)
array2 <- array(c(vector1,vector2),dim = c(3,3,2))
# create matrices from these arrays.
matrix1 <- array1[,,2]
matrix2 <- array2[,,2]
# Add the matrices.
result <- matrix1+matrix2
print(result)


list_data <- list("Red",  c(21,32,11), TRUE, 51.23)
print(list_data)


# Create a list containing a vector, a matrix and a list.
list_data <- list(c("Jan","Feb","Mar"), matrix(c(3,9,5,1,-2,8), nrow = 2),list("green",12.3))
names(list_data) <- c("1st Quarter", "A_Matrix", "A Inner list")
print(list_data)


# Add element at the end of the list.
list_data[4] <- "New element"
print(list_data[4])
# Remove the last element.
list_data[4] <- NULL
# Print the 4th Element.
print(list_data[4])
# Update the 3rd Element.
list_data[3] <- "updated element"
print(list_data[3])


d <- data.frame(name=c("张三", "李四", "王五"), age=c(20, 21, 22), height=c(180, 170, 175),stringsAsFactors=FALSE)
print(d)


# Create the data frame.
d.data <- data.frame(
  d_id = c (1:5),
  d_name = c("Rick","Dan","Michelle","Ryan","Gary"),
	salary = c(623.3,515.2,611.0,729.0,843.25),
	start_date = as.Date(c("2012-01-01","2013-09-23","2014-11-15","2014-05-11","2015-03-27")),
	stringsAsFactors = FALSE)
result <- data.frame(d.data$d_name,d.data$salary)
print(result)

