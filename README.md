# Predictive model for dengue cases and relative risk using climatic variables, Google Trends and campaigns

This repository contains the code used in the article *article*.

## Structure

It contains the following folders:

+ download_data: files containing data extraction from satellites.

+ processing: files that aggregate the extracted data so that each row is a year-week-canton. It contains four types of files:

    + Interpolation for data (linear, time-series or spatial) for missing data or data that is not weekly.

    + Calculating the lags for each variable.

    + One file with in which the relative risk is calculated.

+ data: the processed data used in this repository. It does not contain all the raw downloaded data files because of storage issues. 

    + raw: the first processing step for the data right after downloading it. It also contains the files used for the EDA report. The full CHIRPS and CHIRTS files without weekly aggregated data are in separate folders.

    + interim: it contains the data that is ready for integration in a big dataset. This means it is aggregated weekly and the rows are year-week-canton. It contains a separate folder that contains the weekly lags for each variable (1 to 8 weeks).
    
    + clean: the full dataset.
