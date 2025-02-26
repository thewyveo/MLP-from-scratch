he <- 5
nums <- c(1, 2, 3, 4, 5)

x <- -10:10
y <- x^2
# plot(x, y)
# hist()

# runif() = generater random number

# vectors
# nums2 <- c(1:3)

# rep(c(2,3), times = 2) = 2 3 2 3
# rep(c(2,3), times = c(2,2)) = 2 2 3 3
# rep(c(2, 3), times = c(2, 3)) = 2 2 3 3 3
# rep(c(2,3), length.out = 6) = 2 3 2 3 2 3
# rep(c(2,3), each = 2) = 2 2 3 3
# rep(c(2,3), length = 5) = 2 3 2 3 2

#seq(from = 1, to = 7, by = 2) = 1 3 5 7

# sum(nums) = 15
# max(nums) = 5
# exp(2) = 7.389056 
# log(8) = 2.0794415
# log(8, base = 2) = 3
# str(he) = num 5
# pi, T, or F are set to their values and can be changed.

# nums[-1] = 2 3 4 5
# nums[c(TRUE, FALSE, TRUE, FALSE, TRUE)] = 1 3 5
# nums > 3 = FALSE FALSE FALSE TRUE TRUE
# nums[nums > 3] = 4 5 

# %in% is used to check if an element is in a vector
# 2 %in% nums = TRUE

# odds <- c(1, 3, 5, 7, 9)
# nums[nums in odds] = 1 3 5

# head(rivers) = 735 470 420 390 320 300
# sort(head(rivers)) = 300 320 390 420 470 735 
# sort(head(rivers), decreasing = TRUE) = 735 470 420 390 320 300
# which(rivers > 700) = 1  7  15  16  20  23  24  25  26  32  38  44  50  63  66  67  68  69  70 71  79  82  83  89  90  98 101 109 114 115 121 131 137 14
# table() returns the frequency of each element in a vector

# Important to know with what data type you are working with
# when examine a new dataset, use str() to see the data types of each column

# data types
#   numeric: x <- 2
#   integer: x <- 2L
#   character: x <- "hello"
#   logical: x <- True or False 
#   complex: x <- 1 + 2i
#   raw: x <- charToRaw("hello")

# data structures
#   vector: x <- c(1, 2, 3)
#   factor: levels are the unique values in the vector
#   rating <- factor(rating, levels = 1:5, labels = c("bad", "poor", "ok", "good", "excellent"))

# NA is used to represent missing values.
#   mean(airquality$Ozone, na.rm = TRUE).
#   na.rm = TRUE removes the missing values from the calculation.

# Data frames
# a number of vectors of the same length
# str(mtcars) = 'data.frame': 32 obs. of 11 variables
# mtcars[3,6] = mtcars[3, "wt"] = mtcars$wt[3] = 21.4 = 6th column of the 3rd row
# mtcars[3,] = 21.4 6 160 110 3.9 2.62 16.46 0 1 4 4 = 3rd row
# mtcars[mtcars$mpg > 25,] = rows where mpg is greater than 25 = 21.4 6 160 110 3.9 2.62 16.46 0 1 4 4
# mtcars[mtcars$carb == 2 | mtcars$gear == 3,] = rows where carb is 2 or gear is 3

# max(airquality$Temp)
# junetemps <- airquality[airquality$Month == 6, "Temp"]
# mean(junetemps)
# mostwind <- which.max(airquality$Wind)
# airquality[mostwind, ]

great_lakes <- data.frame(
  name = c("Huron", "Ontario", "Michigan", "Erie", "Superior"),
  volume = c(3500, 1640, 4900, 480, 12000), # km^3
  max_depth = c(228, 245, 282, 64, 406) # meters
)

# Reading data from a file
# data <- read.csv("file.csv")
# data <- write.csv(mtcars, "mtcars_file.csv", row.names = FALSE)
# getwd() = get working directory

normtemp <- read.csv("http://stat.slu.edu/~speegle/data/normtemp.csv")
str(normtemp)

# Packages
# HistData
# to install a package, use install.packages("HistData")
# to load a package, use library(HistData)
library(HistData)
# head(MASS::immer) = :: to specify the package

# rm(list = ls()) = to delete all objects