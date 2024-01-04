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
# install.packages("tidyverse")
library(tidyverse)

# Sept 7, 2023
# version 3.1
# FIXED: temp means intervals
#        location: func_calc_timestep_means.R

# Sept 5, 2023
# version 3.0
#
# loop offset
# save output to different folders
# save images
# no longer need offset...will iterate through it
# NEW PARAMETER: onset




duration <- 30
onset <- 52 # new! the starting date
# onset the # of days past March 10 to begin analyzing

# Set correlation method version
version <- 3
# 1 = dT
# 2 = exp(dT)
# 3 = exp(dT)/cone


# read in the data
source("readin_data.R")

# load the functions which will be used below
source("function_calc_timestep_means.R")
source("function_dT.R")
source("function_r.R")

original_dir <- getwd()

offset <- duration - 1
for (i in (0:offset)) {
  newdir <- glue("output__duration{duration} onset{onset} offset{i}/")
  if (!dir.exists(newdir)) {
    if (!dir.create(newdir)) {
      cat("Failed to create directory:", newdir, "\n")
      next # Skip to the next iteration of the loop
    }
  }

  # IMPLEMENT FUNCTION TO CALCULATE TIMESTEP MEANS
  temp_means <- timestep_means(duration, onset + i)

  # order temp_means by timestep, then year to get it ready to calculate dT
  temp_means <- temp_means[
    with(temp_means, order(Group.1, year)),
  ]

  temp_filename <- paste0(newdir, glue("temp_means__duration{duration} offset{i}.csv"))
  write.csv(temp_means, temp_filename, row.names = FALSE)

  # IMPLEMENT FUNCTION TO CALCULATE dT
  #   output is exp_dT
  exp_dT <- calc_dT(temp_means, newdir)
  expdT_filename <- glue("{newdir}/exp(dT)__duration{duration}_offset{i}.csv")
  write.csv(exp_dT, expdT_filename, row.names = FALSE)

  # IMPLEMENT FUNCTION TO CALCULATE r
  # calc_r(dT,cone,version)
  # 3 = exp(dT)/cone
  # 2 = exp(dT)
  options(scipen = 999)
  r_table <- calc_r(exp_dT, cone, version, newdir)

  # Save r_table table as '.csv'
  #    file is saved to same directory as this script
  #    row.names = FALSE <= removes the number of the row

  r_fname <- glue("{newdir}/r_value__duration{duration}_offset{i}.csv")
  write.csv(r_table, r_fname, row.names = FALSE)
}

# cformat the r output into one file
#   output is saved to a .csv file called r_duration{duration}.csv
source("format_r_output.R")
