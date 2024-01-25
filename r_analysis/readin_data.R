#  CONE DATA
#    read in cone data

library("readxl") # used to read in data
library(lubridate)

tsuga <- read.csv("Cone Crop for Delta-T calculations v.1 2.csv")
tsuga_sites <- tsuga$Code

# Select the year columns of the cone crop data; transpose
# the table so the years are rows, then add 0.5 to each cone crop
cone <- t(tsuga[, 19:ncol(tsuga)]) + 0.5
colnames(cone) <- tsuga_sites

#  TEMP DATA
#   read in temperature data
path <- paste0(getwd(), "/daily weather data 1981-2014.xlsx")

#   read in col names
headers <- read_excel(path, skip = 10, col_names = TRUE) %>%
  names()
