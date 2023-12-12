############<<<

# SCRIPT TO CALCULATE R values
# Dataframes used
# - dT
# - cones
#  METHOD
#     use Pearson corr to calculate r value, correlation coefficient (r)
#       between dT and the current month's cone values
#  OUTPUT
#     r_table:
#              sites x month
#              the r values!
#     csv files:
#              {outputfolder}/r_value__duration{duration}_offset{offset}.csv
#              {outputfolder}/r2_value__duration{duration}_offset{offset}.csv



# Calculate r values 3 different choices
# v1: delta t
# 
# v2:exp(delta t)
# The predictor for the correlation is now exp(delta t)
# Consider that the seed crop records are strongly right skewed but the weather 
# inputs are by definition gaussian (i.e. an additive process). The internal 
# hormonal response to the gaussian input must be exponential.
# 
# v3: [exp(delta t)] / previous years crop size
# The predictor is [exp(delta t)] / previous years crop size.
# Obviously, you cannot do this correlation for the very first year. The resource 
# dynamics within the tree will not permit two big crops in a row no matter what 
# delta t is...Big years are exhausting; the annual growth ring during a big crop 
# is always small.
# 
# We use Pearsons Correlation to calculate r values


library(glue)

calc_r <- function(dT,cone,v,outputfolder){
  # list of timesteps
  timesteps = unique(sort(dT[,'#timestep']))

  # list of sites
  sites = colnames(dT)[3:ncol(dT)]

  # create dataframe to hold correlation values
  r_table=data.frame(matrix(nrow=length(sites),ncol=0))

  # calculate for each timtesteps, t
  for (t in timesteps) {
  
    # for this specific timestep (t)
    # create holding zone for each of the r's for the #S sites
    r_timestep=c()
  
    # calculate for each site, s
    # calculate the r's (as a list) for each of the #S sites
    
    for (s in sites){
      
      #s = sites[S]
    
      # INDEX SPECIFIC dT TIMESTEP (dT_t) for SPECIFIC SITE (s) 
      dT_t = dT[which(dT$'#timestep'==t),s]
      dT_t = dT_t[-length(dT_t)]
    
      # CONE DATA FOR SPECIFIC SITE (s)
      # cone (years x sites)
      #
      #   years 1983 - 2014 (index = 2)
      cone_site=cone[3:nrow(cone),s]
    
      #
      # 10-2-2023 edit
      #  no longer used in code
      #   8-2-2023 edit
      #   cone data for years 1981 - 2013
      cone_site_prev = cone[2:(nrow(cone)-1),s]
      #   8-2-2023 edit
      # 10-2-2023 edit

    
      # calculate r with function cor.test()
      # using the pearson method???? maybe different methods>>>
      
      # 8-2-2023 edit
      #    divide dT by previous year's cone value
      # dT_t/cone_site_prev
      # 
    
      # version 2
      if (v==2){
        corr=cor.test(dT_t,cone_site, method='pearson')
      }
      
      # version 3
      else if (v==3){
        corr=cor.test(dT_t/cone_site_prev,cone_site, method='pearson', use='complete.obs')
      }
      # version 1
      else{
        corr=cor.test(log(dT_t),cone_site, method='pearson')
        }
      r_estimate=as.numeric(corr$estimate)
    
      # saving and storing the r's
      r_timestep[s]=r_estimate
    
      #    print(glue('
      #  
      #  plot:{plot_name} | month: {m}
      #             '))
      #  print(corr)
    }
    #print(r_timestep)
  
    # attach the newly calculated r's to the r_table :)
    r_table=cbind(r_table,r_timestep)
  }
  
  # MAKE r_table look pretty
  #   rename the cols as the months
  colnames(r_table)=timesteps
  rownames(r_table)=sites
  
  # create table for R^2 values
  r2=r_table^2
  
  #   attach a new column of the speices
  r_table$sites=sites
  
  #   sort r_table by species
  #r_table <- r_table[order(r_table$sites),] 
  
  return(r_table) 
}