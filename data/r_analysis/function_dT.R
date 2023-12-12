# CALCULATE dT
#   calculate mean by timestep group ($Group.1)
####>>>>>
####>#   Code to calculate dT
#     - iterate through the rows (#timestep * #year x site)

# FUNCTION
# partition timesteps and calculate timestep means

library(dplyr)
library(reshape2)
library(glue)
library(MASS)
library(stringr)
library(tidyr) # used to reshape (spread) data into wide format

calc_dT<- function(temp_means,outputfolder){
  
  dT=data.frame(matrix(nrow=nrow(temp_means),ncol=ncol(temp_means)))
  colnames(dT)=c('dYear','#timestep',c(colnames(temp_means)[2:(ncol(temp_means)-1)]))
  for (i in 1:(nrow(temp_means)-1)){
  
    # quick check to make sure we are calculating dT for the same timestep
    if (temp_means$Group.1[i+1]!=temp_means$Group.1[i]){
      #print(glue('new timestep:{temp_means$Group.1[i]}>{temp_means$Group.1[i+1]}'))
      next                   # if the months are different, just skip to the next!
    }
  
    # if timesteps are the same, we calculate the difference
    else{
      # create a label for which years we are dT-ing
      dYear=paste(temp_means$year[i+1],'-',temp_means$year[i])
    
      # store the values in dT table
      dT[i,1]=dYear                     # year difference, ex ( 2001-2000 )
      dT[i,2]=temp_means$Group.1[i]         # current timestep,   ex ( "03-10 to 03-16" )
      dT[i,3:ncol(dT)]=temp_means[i+1,2:48]-
        temp_means[i,2:48]
      # difference in temps, dT value
      
    }
  }
  
  ########## RE ORDER COLUMN NAMES
  original_colnames <- colnames(dT)[3:49] # Extract original column names

  # Remove the last 3 characters from each column name
  col_letters = gsub("[[:digit:]]", "", original_colnames)
  modified_colnames <- substr(col_letters, 1, nchar(col_letters) - 3)
  
  # Sort the modified column names alphabetically
  sorted_indices <- order(modified_colnames)
  #print(modified_colnames[sorted_indices])
  
  # Create a new data frame with columns sorted based on the indices
  dT <- dT[, c("dYear", "#timestep",original_colnames[sorted_indices])]
  
  ###########
  
  # calculate exp(dT) to use in further calculations
  exp_dT = dT # copy dT over
  exp_dT[,3:ncol(exp_dT)] = exp(dT[,3:ncol(dT)]) # exp(dT)
  
  return(exp_dT)
}