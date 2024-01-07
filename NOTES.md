# Notes

## function_calc_timestep_means.R

This file generates time intervals which will be correlated with cone crops,
using `daily_weather_data_1981-2014.xlsx`.

- L23: `read_excel` skips 20 rows at the beginning of the weather data, skipping 9
  rows of usable data for each year.
- L30: assign `start_date` the value `yyyy-3-10`, where `yyyy` is the year of
  the weather data from the current sheet.
- L34: Get the temperatures from March 10 onward for the given year
- L37: Get the temperatures from March 10 onward; drop nan values, group by
  site, and "spread" `tmean` on each site (?) possibly broken, doesn't match
  any call signature for that function
- L52: Get the number of intervals for the given onset and duration
- L58: Get the temperatures from March 10 onward
- L63: Create a vector of indices which will be used to index into the data
  which correspond to the start of the date ranges
- L64: Create a vector of indices which will be used to index into the data
  which correspond to the end of the date ranges
- L68, L69: Extract the year for the date range starts and ends
- L85: Compute the mean of the temperatures for the date ranges. If/else
  branches do the same thing. On the first sheet, a new data frame is
  constructed; otherwise the new temp means get concatenated to the existing
  data frame.
