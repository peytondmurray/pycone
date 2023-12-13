# This script takes in raw data and preprocesses it for use in the model.
#
# daily_weather_1981-2014.xlsx

#Libraries
import pandas as pd

# working variables
yrRange = range(1981,2014)

print(yrRange)

# Read in raw data
weatherDF = pd.read_excel('raw_data/daily_weather_1981-2014.xlsx', sheet_name="1981") 