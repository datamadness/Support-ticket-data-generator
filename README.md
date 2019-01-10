# Support ticket data generator

This code will allow you to input few numeric parameters and quickly generate custom support ticket datasets that reflect what you could expect in real world business operations.
#### Feature Overview

* A python function to generate one year worth of support data with arbitrary amount of records(call repeatedly for x years worth of data)
* Generates varied, but statistically relevant number of support tickets for each day of the year
* Effects of business days vs weekends
* Capable to simulate an impact of arbitrary number of busy seasons through the year(e.g. Christmas in retail or tax periods in accounting)
* Simulates realistic, but easy to control statistical distribution of logged time for each support ticket / case.
* Control number of customer accounts to capture desired support volume vs customer base size

#### Input parameters summary

Daily ticket volume through the year controls:

* Seasonal peaks
* Weekend volumes
* Daily volume range (population mean, approximate min and max outliers)

Effort / logged time per ticket controls:

* Median
* Minimum logged time
Advanced / Optional:
* Skewness factor
* Degrees of freedom
* Non-centrality (impacts variance and kurtosis)

Visit [this blogpost for more details](https://datamadness.github.io/Support-Data-Generator)