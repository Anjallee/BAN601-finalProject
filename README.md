# Interactive EDA Dashboard (Streamlit)



A fast, practical EDA and cleaning tool for CSV Data files.

It lets you explore distributions, correlations, outliers, and run a one-click Clean Pipeline with sensible defaults and selectable categorical imputation.



## Features

- Multi‑CSV selection from the current working directory

- EDA tabs: Overview, Distribution, Data Quality, Correlation

- Clean Pipeline (All‑in‑One): Operations performed in sequence below

- Categorical standardization (trim/whitespace collapse/lowercase)

- Date parsing (acceptance threshold > 60%)

- Numeric parsing (acceptance threshold > 70%)

- Outlier flagging (3×IQR) – report only

- Numeric median imputation

- Categorical imputation strategy (selectable):

- None (leave as NA)

- Constant: 'Unknown'

- Mode per column

- Group‑aware (mode within a grouping, then `'Unknown'` fallback)

- Duplicate removal (full-row, keep = 'first')

- Save cleaned data as “clean_<original>.csv” in the same folder



## Quick Start

1. Install dependencies (Streamlit, pandas, numpy, seaborn, matplotlib, scipy) as listed in requirements.txt file..

1. Place your CSVs in the app’s working directory.

1. Run the app: streamlit run Full_EDA_project.py



## Raw Data Insights:

For boston_housing.csv data file the Dataset Statistical Overview and the First 5 row raw data are as shown below.

![image](image_1.png)

The Data Quality Analysis on the above data shows the details in terms of Unique Values, Missing Values and Missing %. Given in table below:

![image](/images/data_quality_table.png)

The following Data Cleaning operations were peformed on the raw data:

- Categorical standardization (trim/whitespace collapse/lowercase)

- Date parsing (acceptance threshold > 60%)

- Numeric parsing (acceptance threshold > 70%)

- Outlier flagging (3×IQR) – report only

- Numeric median imputation

- Categorical imputation strategy : Group‑aware (mode within a grouping, then 'Unknown' fallback)

- Duplicate removal (full row, keep = 'first')

Post above operations clean_boston_housing.csv is saved in the app folder.



## Cleaned Data Insights:

The Data Quality Analysis shows the data has been successfully cleaned as can be seen in the Column Summary (compact) table below.

We can clearly see across all columns, the Unique Values, Missing Values and Missing %, are zero.

![image](image_3.png)

EDA on Cleaned Data:

## Histogram distribution on Age column is shown below:





![image](/images/age_histogram.png)

NOTE: Histogram distribution can be obtained for any column selected via the drop down (eg.. lstat, medv, tax, rad etc.)



| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| Value | 506 | 69.75 | 26.75 | 2.9 | 48.63 | 77.7 | 92.9 | 100 |


Key Business Insights (from Age Distribution Histogram):

Most neighborhoods show high values of age (75–100), meaning a large proportion of houses were built before 1940.

This indicates a housing market dominated by older properties.

Older homes create strong demand for renovation, maintenance, and modernization services.

Real estate developers may find redevelopment opportunities in these areas.

Insurance and maintenance costs may be higher due to aging infrastructure.

Only a small portion of neighborhoods have newer housing, suggesting limited recent development.



## Pearson Correlation:



![image](/images/correlation_matrix.png)

NOTE: Strongest correlation between: rad and tax

Key Business Insights (from the correlation matrix):

Number of rooms (rm) has the strongest positive correlation with house price (medv), meaning larger homes tend to be more valuable.  NOTE: medv = Median House Value.

Lower-status population percentage (lstat) has a strong negative relationship with prices, indicating neighborhoods with better socioeconomic conditions command higher property values.

Education quality (ptratio) negatively correlates with prices, suggesting homes in areas with better schools (lower pupil–teacher ratios) are more desirable.

Higher property taxes (tax) and pollution levels (nox) are associated with lower housing prices.

Environmental and neighborhood quality factors significantly influence real estate valuation alongside property size.

Overall, location quality, education infrastructure, and property size are the key drivers of housing prices in this dataset.











