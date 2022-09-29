# PREDICTING USED CAR SALES PRICES<a name="top"></a>
![]()

by: Dan Churchill

<p>
  <a href="https://github.com/DanChurchill" target="_blank">
    <img alt="Dan" src="https://img.shields.io/github/followers/DanChurchill?label=Follow_Dan&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
[[Steps to Reproduce](#reproduce)]
___



## <a name="project_description"></a>Project Description and Goals:
The purpose of this project is utilize a database of used auto sales to construct a model that can predict their sale price. The datasource includes the Manheim Market Report price, an industry standard for pricing automobiles. Can that model be beaten?

Goal 1: Build a model using only the features of the vehicle to predict sales price.

Goal 2: Build a model using the MMR estimate along with the auto's features to improve upon the MMR model.

Goal 3: Try some tools I haven't previously used including Skim, Pipeline, TargetEncoder, and XGBoost.


[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 


### Project Outline:
- Acquire and prepare data from a locally saved CSV originally obtained from <A href="https://www.kaggle.com/code/desalegngeb/auctioning-used-cars-what-matters-most/data?select=car_prices.csv">Kaggle.com</A>
- Establish a baseline RMSE
- Perform data exploration to determine what features will be usefull for modeling
- Train two different linear regression models without using the MMR predictions
    - Make predictions and evaluate.  Did we beat MMR?
- Train two linear regression models including the MMR predictions
    - Make predictions and evaluate.  Did we improve on MMR?
- Choose the model with that performed the best and evaluate that single model on the test dataset
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

[[Back to top](#top)]
***

## <a name="dictionary"></a>Data Dictionary  

| Target Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| sellingprice | the actual selling price of the used vehicle | int |


---
| Feature | Definition | Data Type |
| ----- | ----- | ----- |
| year | The model year of the vehicle| int |
| make | The manufacturer of the vehicle | string |
| model | The model of the vehicle | string |
| trim | The trim level of the vehicle | string |
| body | The bodystyle of the vehicle | string |
| transmission | The type of transmission in the vehicle | string |
| vin | the vehicle identification number of the vehicle | string |
| state | two-digit state/province code where the sale occurred | string |
| condition | subjective 1.0-5.0 condition rating of the vehicle | float |
| odometer | the mileage of the vehicle | float |
| color | the exterior paint color of the vehicle | string |
| interior | the interior color of the vehicle | string |
| seller | the seller of the vehicle | string |
| mmr | the manheim market report valuation of the vehicle | int |
| sellingprice | the actual selling price of the vehicle | int |
| saledate | the date the vehicle was sold | string |

[[Back to top](#top)]

***

## <a name="wrangle"></a>Data Acquisition and Preparation

Data is acquired from the Codeup Database server using an SQL query within the modular function wrangle_zillow located in the wrangle.py file.  This returns 43182 rows and 31 columns split into train, validate, and test dataframes in a 60% / 20% / 20% ratio.

Preparation is performed in the wrangle function prior to splitting consisting of the following:

- Converts the transaction dates to Datetime format, and removes a row with a 2018 property erroneously saved in the 2017 table
- Converted the fips data to the actual name of the county
- Removes the leading digits and decimal portion of the census tract
- Converts nulls to zero where applicable in binary columns
- Removes rows where unit count is greater than one, since these are likely erroneosly categorized as single-family
- Deleted rows where a tax delinquency exists, as this could impact the tax values in ways we do not understand
- Converted year built to age by subtracting the value from current year (2022)
- Createed binary categorical columns for homes with greater than three bedrooms, and three to five garage parking spaces
- One-hot encoded county, and bedroom count values
- Renamed columns for clarity and to streamline programming
- Rows with outliers are removed



[[Back to top](#top)]

![]()


*********************

## <a name="explore"></a>Data Exploration:

### Locate properties
The first step was to use the fips value to identify the county each property was located in.  There were three values: 6037, 6059, 6111 corresponding to Los Angeles County, Orange County, and Vetura County repsectfully, all in California.

### Exploring Tax Value
Next we look at the distribution of the target variable, Tax Value.  The values appear somewhat normal, although right-skewed.  Values appear highest in Los Angeles county, followed by Orange County and Ventura County.  Testing using variance testing allows us to reject the null hypothesis that values are the same in all three counties.

### Exploring Specific Location
Exploring the values graphically on a map show that there are clusters of higher and lower valued homes within each county.  For this reason the tract column is used in modelling.  Ideally, this would be used as a categorically, but there are too many unique values to categorically encode.

### Correlation of Numerical Features
Using a correlation matrix we see that square footage is the most correlated to tax value, followed by bedrooms and bathrooms.  However, because bedrooms and bathrooms are highly linearly correlated to square footage we do not want to use them directly in the model as a numerical value.  I researched if an adjusted square footage was feasible, but according to the National Association of Homebuilders the percentage of square footage allocated to bedrooms and bathrooms remains constant at 40% irrespective of home size [(Source)](https://bestinamericanliving.com/2016/08/where-builders-place-their-space-2/)
  </a>
</p>  For this reason I explored ways to use bedrooms and bathrooms categorically.

### Exploring Number of Bathrooms
Using a box plot we look graphically at the how the number of bathrooms affects property value.  We see that the effect appears linear up to 3.5 bathrooms as which point it jumps sharply and levels off.  We created a categorical variable to capture properties that had more than 3 bathrooms for use in modelling.  And looked at the populations of each subset graphically.  While three or less bathrooms were consistant with the mean, properties with more than three bathrooms had a significantly higher property value.  This was confirmed by a T-Test where we rejected the null hypothesis that the values were equal.  

There were similar observations in garage size and a similar category was created for garages that had three to five parking spaces.  Bedroom count showed no similar trend, so we used it as a categorical variable. 



### Takeaways from exploration:
- We've identified that location is vital to property value
- Square footage, bedrooms, and bathrooms are all key drivers of property value, but are correlated to each other and must be used differently
- High number of bathrooms is significant, as are large garages which can be used as categories

[[Back to top](#top)]

***

## <a name="model"></a>Modeling:

#### Modeling Results
| Model | RMSE on train | RMSE on validate | R2 score |
| ---- | ---- | ---- |---- |
| Baseline | $233,115.06 | N/A | N/A |
| Linear Regression (OLS) | $198,418.25 | $199,021.70 | 0.2816 |  
| LassoLars | $198,424.85 | $199,017.21 |  0.2816 |
| Tweedie Regressor | $198,944.70 | $199,738.47 | 0.2764 |

 


- The LassoLars model performed slightly better than the OLS and Tweedie Regressor models


## Testing the Model

- Tweedie Regressor model used on Test data

#### Testing Dataset

| Model | RMSE on test | R2 score |
| ---- | ---- | ---- |
| Tweedie Regressor | $198,604.47 |  0.278 |

[[Back to top](#top)]

***

## <a name="conclusion"></a>Conclusion and Next Steps:

- We created a tax value predictor that beat the baseline by $35,000

- The model performed significantly better when the dataset was restricted to a more narrow set of data

- Location was the largest driver of tax value, followed by square footage

- I attempted to create a function that created a dictionary of models trained for each of the most popular tracts
    - Consolidating results and evaluating proved too difficult to implement in the time given

[[Back to top](#top)]

*** 

## <a name="reproduce"></a>Steps to Reproduce:

You will need your own env.py file with database credentials then follow the steps below:

  - Download the wrangle.py, explore.py, modeling.py, and final_report.ipynb files
  - Download the FIPS, points, scale, and tract JPEGs for visualizations
  - Add your own env.py file to the directory (user, host, password)
  - Run the final_repot.ipynb notebook

[[Back to top](#top)]
