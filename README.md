# pycone

## Variables

Response variable: Cone crop

explanatory variable: Calendar day and delta-t value

## Cone production records

We used seed production data for 10 mast-seeding species located in the
Pacific Northwest from Franklin and Schulze (2023) annual ovulate cone
production surveys. These surveys included the most consistent
methodology of surveys and had the longest records available for several
genera within the Pinaceae family. Their surveys were conducted for
*Abies* spp. (*A. amabilis, A. concolor, A. grandis, A. lasiocarpa, A.
magnifica, A. procera*), *Pinus* spp. (*P. engelmannii, P. lamberti, P.
monticola*), and *Tsuga* spp. (*T. mertsiana*) (Franklin and Schulze,
2023). A total of 61 plots in 37 locations were collected in nine
National Forests in Washington and Oregon from 1959 to 2023. These sites
are part of the LTER network. [At each site, a visual count was made of
cone production in each of a number (20-30) trees in a stand of one tree
species (Franklin and Schulze 2023)]. Many of these records have
a year or sequence of years marked as N/A (no data collected). Our
analysis only included the longest non-NA portion of any record, leaving
a total of 36 records for our analysis (10 *A. amabilis,* 2 *A.
concolor, 3 A. grandis, 1 A. lasiocarpa, 3 A. magnifica, 6 A. procera*),
*Pinus* spp. ( *1 P. lamberti, 2 P. monticola*) and *Tsuga* spp. (7 *T.
mertsiana*). Thus while the bulk of these records span >40 years, the
record portions we used are typically more in the range of 15-30 years.

## Climate data

For all analyses, we used mean daily air temperature for March 1st
through October 1st for a 33 year span (1981-2014) to match the
available cone crop production records from the Franklin, J.F. and M.D.
Schulze dataset. Temperature data from the nearest weather station was
taken from the PRISM Climate Group at Oregon State University; see
[https://prism.oregonstate.edu/](https://prism.oregonstate.edu/).
Site location data was extracted from the Franklin and Schulze (2023)
dataset and uploaded to the PRISM explorer for time series values with
individual locations. The proximity of the PRISM weather station and
site location did not exceed 4 kilometers.

## Statistical Analysis

Prior to our analysis, we calculated Delta-t using daily temperature
data for each year for the months of March through October. For the
genera with a 2 year seed development time (*Abies and Tsuga*), we
calculated Delta-T from daily temperature in year T2 minus daily
temperature for T1. Then we modified the onset (start day) and duration
(length of days) so that Delta-T could be calculated for different
intervals. For example, onset 1 duration 30 would calculate the Delta-T
values for March 1st through March 30th given the daily temperature
means for 1983 minus the daily temperature means for 1982 (T2-T1).

Then, Pearson's correlation was calculated between the Delta-T values
for each group of dates and the cone crop for each species annual cone
crop record. For example, Delta-T for 1983-1982 predicts the ideal
temperature cue for the 1984 cone crop. This was done in two ways:
version 2 was using the exponential of Delta-T and version 3 was the
exponential of delta-t/cone crop.

## Background:

The Delta-T (ΔT) model uses both T ~n-1~ and T ~n-2~ to predict cone
production. This model can be used to determine the temperature cue that
initiates reproductive buds by the difference in summer temperature one
year before the crop (T ~n-1~) and the temperature in the preceding
summer (T ~n-2~ ) i.e ΔT= T ~n-1~ -T ~n-2~ (Kelly, 2013; LaMontagne
2020). The predicted cone crop size will respond to the magnitude of the
difference between the two earlier summers (Kelly 2013; LaMontagne 2020;
Vacchiano 2017) with large crops corresponding to large positive values
of ΔT

We already calculated delta-t using month values and then correlated
using the Pearson correlation test. Now, we see which month has the best
correlation between delta-t and the cone crop values for that year. We
speculated. We said that the onset date is the Julian calendar date and
the assumed calendar date is 30 days but nature doesn't do that. We can
make the assumption that it may be a certain month but we don't know
After delta-t for the month, we will see which week has the best
correlation and go through all of the weeks in that month with staggered
on set dates (May 15- September 8th). We do exactly what we did before,
we do delta-t weekly not monthly and we will do that for all of the
weeks for May 15- september 8th. About 100 different onset date. Repeat
the process with two weeks then three weeks and then., one month again
and test the four different onset dates.

## Objective:

-   Define the months or dates for specific species.

-   Doesn't matter what we find because no one has done this

-   Organizing the data;

    -   Cone crop data- (this is the median values)

        -   Only use consecutive values: that means exclude the values
            that are NA

        -   Isolate the non-NA values for each species. Longest string
            of non-NA

        -   Run both fragments

            -   Ex : (Duration) 30, 20, 18, 6

    -   Weather data-

        -   relevant years that match with the consecutive cone crop
            data

## David Ranting at Emily and notes for the original program:

-   Start with all the temperature dates as daily values and set
    duration

-   First will be one week starting on may 15 to may 21st inclusive is 7
    days so the very dist time you try the code you do it based on mean
    temp over the next seven days

-   Now for each year you have mean temp from may 15 to 21 then
    correlation with seed crop size

-   Then run it again and start with may 16 and run for seven days → may
    17th → do until final onset date for september 8th

-   Different onset with one week duration. Then do it for a two week
    duration for 14 days inclusive and then get correlation. And keep
    going until two weeks before September 18ths.

-   Then do it again for three week durations from may 16

-   Then one month but can only go until June.

-   Then look at all correlations which species had the most consistent
    correlation and which onset and correlation

## Delta-T model explaination:

**Notes from Emily for the Delta-T function in R**

-   Didnt use any main functions, only used ggplot for graphing,
    functions used are ones that are custom functions.

-   Reading in the temperature data: storing all of the data in file
    temp1, then temp 1a takes specific col from temp 1.

-   Delta T; Daily temperature for each year. Depending on when you
    wanted to start March 10 to October 1st. 8x30 = 240 days. Depending
    on the duration and interval you were looking at, 30 days, 1 week, 2
    weeks. Code assigns groups to each of the days, if one week then
    March 10,11,12 then 7 of those days are assigned group 1. Across the
    year, the group 1s were averages and that would be the temp mean for
    1981, then it repeats for 1982 and 1983 and so on. Youd have to do
    this for each group of days for all of the years (1982-2014). This
    is taking the temperature mean difference for the years. Calculate
    time step means: spits out the mean and another function takes in
    the means and takes difference between the years. Labeled col and
    order.

## Cross check dates with low cone production: Notes from David

-   there is no unanimity on the delta t signal with regard to onset or
    duration because the cone crops themselves are not well
    synchronized. THAT had never occured to me as a potential problem. .
    . nonetheless with amabilis we can pick out one group of sites whose
    crop records are highly correlated with one another. It is true we
    have no clue why they are poorly correlated (something local. . .
    hail storm, local insect infestation. . .) but why not go with the
    highly correlated ones and see with that group only: how ell does
    delta t predict and what are the onset and duration.

-   if you do the correlation matrix for the other species, then we can
    decide on a suitable cut-off; eg all conspecific records with a crop
    correlation of r>0.8 will be used.

-   then you feed the selected records into the program as a large
    single record. eg we have 4 records totaling 120 years and that goes
    in as a single record (even though years are being repeated). that
    is what emily is supposed to be working on.

-   I picked two sites well separated by latitude and correlated the
    first ten days of march for each. r=0.9. but the cone crop
    correlations show quite a bit of difference unrelated to latitude.
    ie presumably unrelated to weather. something other than weather is
    going on here. eg 1985 was a great crop at all sites for amabilis
    except ABAM36 where there were ZERO cones. ABAM35 should have a big
    crop in 91 but it doesnt. likewise, 97 is a great crop year for some
    amabilis sites but not for others. lots of examples. --this cant be
    due to weather; it is too similar for nearby sites. I think we will
    conclude there are limits to what delta t can do because there are
    non-weather drivers we dont understand. nonetheless we can still
    offer them something (onset/duration for each species that we expect
    to have an average r of about

## What we need now...

Picture a species with 8 records. Presently, each record has a set of
cone crop values and delta t values, and we run each record separately.
Is there a way to combine all the 8 records into a single record and
then run the program on that? Ultimately we want to know the best onset
and duration for the species (not just for each record). [In other
words, the records are currently broken up by site but instead we want
to break them up by just the species. i.e all records that have ABAM for
example, would be grouped together instead of separated by their
location site.]{.mark}
