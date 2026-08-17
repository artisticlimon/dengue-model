# Predictive model for dengue cases and relative risk using climatic variables, Google Trends and campaigns

This repository contains the code used in the article *article*.

## Structure

It contains the following folders:

+ download_data: files containing data extraction from satellites.

+ processing: files that aggregate the extracted data so that each row is a year-week-canton. It contains four types of files:

    + Interpolation for data (linear, time-series or spatial) for missing data or data that is not weekly.

    + Calculating the lags for each variable.

    + One file with in which the relative risk is calculated.
  
+ results: notebooks used to train the models, calculate NRMSE and its bootstrap intervals, and generate tables and figures.

+ full_data: full curated data used for all computations.
