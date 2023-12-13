# This script takes in raw data and preprocesses it for use in the model.
#
# daily_weather_1981-2014.xlsx

#Libraries
import pandas as pd


#pull_Weather_Data(inputFile, startYr, endYr)
# working variables for testing
inputFile = 'raw_data/daily_weather_1981-2014.xlsx'
startYr = 1981
endYr = 2014

# Intialize list for reading in sheets from excel file and creating range of years for looping
yrRange = range(startYr,endYr+1)
weatherYrZ = []

# Looping through yrRange to read in each sheet as a dataframe into list weatherYrZ
for year in yrRange:
    df = pd.read_excel(inputFile, sheet_name=str(year), skiprows=10)
    df['Year'] = year  # Add a new column 'Year' filled with the current year
    weatherYrZ.append(df)


# Merge all df into one big df with data from all years ~2.5 million rows
bigWeather = pd.concat(weatherYrZ, ignore_index=True)
#Had odd empty rows so we drop them
bigWeather = bigWeather.dropna(subset=['Name'])

#output to file for QAQC, return for function calling
bigWeather.to_csv('bigWeather.csv', index=True)
#return bigWeather