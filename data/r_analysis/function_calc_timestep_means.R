# FUNCTION v2
# TIMESTEP renamed to DATE RANGE
# partition timesteps and calculate timestep means

library(tibble)
library(ggplot2)
library(dplyr)
library(reshape2)
library(glue)
library(MASS)
library(stringr)
library(tidyr) # used to reshape (spread) data into wide format
library("readxl") # used to read in data
library(lubridate)
library(stringr)

timestep_means <- function(duration, onset) {
  for (i in 1:34) {
    sheet_numb <- i

    # READ IN DATA
    #   temp1 is the full dataset
    temp1 <- read_excel(path, sheet = sheet_numb, skip = 20, col_names = headers)

    #   temp1a, reduced temp1 dataset
    temp1a <- data.frame(temp1$Date, temp1$Name, temp1$`tmean (degrees F)`)
    colnames(temp1a) <- c("date", "site", "tmean")

    Year <- year(temp1a$date[1])
    start_date <- ymd(paste0(Year, "-", 3, "-", 10))

    #   temp1b, further reduced dataset
    #           select dates from March 10 and on
    temp1b <- temp1a[temp1a$date > start_date - 1, ]

    #   temp, reshape dataframe to be (groups x site)
    temp <- temp1b %>%
      na.omit() %>%
      group_by(site) %>%
      spread(site, tmean)

    # CALCULATE TIMESTEP MEANS
    # input: duration and onset [days]
    # function: aggregate to find means by timestep group

    # total number dates in temp dataframe
    n <- nrow(temp)


    # Assign DATE_RANGE groups to temp values
    ## m = number of date_range groups
    m <- (n - onset) %/% duration
    ## end = index of last element in last date_range group
    end <- n

    # Subset of temp without the onset
    ################## Sept 7, 2023 FIX
    temp_time <- temp[(onset + 1):(n), ]
    ################## (end) Sept 7, 2023

    # Create date_ranges: 'start - end'
    ## index for START and END
    date_i_start <- seq(1, end - duration, by = duration)
    date_i_end <- seq(duration, end, by = duration)

    # Extract the year from the daily weather data
    ## index the value for START and END
    date_start <- substr(temp_time$date[date_i_start], 6, 10)
    date_end <- substr(temp_time$date[date_i_end], 6, 10)

    ## concatenate the START and END to a string range
    date_ranges <- data.frame(date_start, date_end[1:length(date_start)])
    colnames(date_ranges) <- c("start", "end")
    date_ranges$range <- glue("{date_ranges$start} to {date_ranges$end}")
    date_range <- rep(date_ranges$range, each = duration)[1:nrow(temp_time)]
    ## attach date_range to the dataframe
    temp_time$date_range <- date_range

    # print(date_range)

    # calculate means by group
    #  output: (dataframe) (total.days/duration x #site) timestep means for all species


    if (sheet_numb == 1) {
      temp_means <- aggregate(
        x = temp_time[, colnames(temp[-1])],
        by = list(date_range),
        FUN = mean
      )
      temp_means$year <- rep(Year, nrow(temp_means))
    } else {
      temp_means_new_sheet <- aggregate(
        x = temp_time[, colnames(temp[-1])],
        by = list(date_range), FUN = mean
      )
      temp_means_new_sheet$year <- rep(Year, nrow(temp_means_new_sheet))
      temp_means_new <- rbind(temp_means, temp_means_new_sheet)
      temp_means <- temp_means_new
    } # end appending tmeans
  } # end sheet reading loop

  return(temp_means)
} # end function
