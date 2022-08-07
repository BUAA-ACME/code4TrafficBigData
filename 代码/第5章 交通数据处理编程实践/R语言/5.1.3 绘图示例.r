curve(x^2, -2, 2)
curve(sin(x), 0, 2*pi)
abline(h=0)
dev.off()

# Create the data for the chart.
H <- c(7,12,28,3,41)
M <- c("Mar","Apr","May","Jun","Jul")
# png(file = "bar_chart.png")
barplot(H,names.arg = M,xlab = "Month",ylab = "Revenue",col = "blue",
main = "Revenue chart",border = "red")
dev.off()

# Create the data for the chart.
v <- c(7,12,28,3,41)
t <- c(14,7,6,19,3)
plot(v,type = "o",col = "red", xlab = "Month", ylab = "Rain fall", main = "Rain fall chart")
# png(file = "line_chart.jpg")
lines(t, type = "o", col = "blue")
dev.off()

# Create data for the graph.
x <- c(21, 62, 10, 53)
labels <- c("London", "New York", "Singapore", "Mumbai")
# png(file = "pie_chart.jpg")
pie(x, labels, main = "City pie chart", col = rainbow(length(x)))
dev.off()