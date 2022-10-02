# PREDICTING USED CAR SALES PRICES<a name="top"></a>
![]()

by: Dan Churchill

<p>
  <a href="https://github.com/DanChurchill" target="_blank">
    <img alt="Dan" src="https://img.shields.io/github/followers/DanChurchill?label=Follow_Dan&style=social" />
  </a>
</p>

## <a name='executive_summary'></a>Executive Summary:
This project attempted to find the key drivers of selling price in Used cars using records of over 500K actual sales.  Then a model was created to try and  outperform the Manheim Market Report (MMR).  While initially unsuccessful, I was able to utilize the MMR value as a feature to improve upon the MMR predictions by over 16%. The improved MMR model could easily be implemented to gain an advantage in the used car market, although it would require an updated data source prior to implementation.

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
The purpose of this project is utilize a database of used auto sales to construct a model that can predict their sale price. The datasource includes the Manheim Market Report price, an industry standard for pricing automobiles. Can that model be beaten and give us a business advantage?

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

Data is acquired from a locally stored csv.  This returns 558,881 rows and 16 columns.

Preparation is performed by the carwash1 function which completes the following:

- Converted nulls in transmission category to 'unknown' 
- Remaining nulls were dropped
- Used string manipulation to standardize values in the make, model, trim, color, interior, and state categories
- Filtered extreme outliers from our target variable, sellingprice

[[Back to top](#top)]

![]()


*********************

## <a name="explore"></a>Data Exploration:

### Investigate Target Variable
The first step was to use plot the distribution of sellingprice.  The data is somewhat of a normal distribution, albeit right-skewed.

### Exploring MMR
Next we look at the MMR and plot it vs actual selling price.  We calculate that the MMR model produces a RMSE of $1670, far superior to the mean baseline of $8757

### Exploring Transmission Type
Exploring the values graphically on a map show that there are differences in the mean selling value of vehicles depending upon what type of transmission is equipped.  We use an ANOVA test and were able to reject the null hypothesis that there is no difference.  This feature will be included in modeling, and the same process was performed on other categorical variables

### Exploring Vehicle Make
Plotting the top 5 highest value makes, and the lowest 5 valued makes shows there is a wide range of values depending upon the manufacturer of the vehicle.  Because we have over 40 makes in our dataset, One-hot-encoding is not feasible.  We will encode using target variable encoding which assigns the make's mean selling price as the value for the encoded column.  This preserves the disparity in values more effectively than one-hot would.  The same encoding will be used on the other categorical values prior to modelling.

### Co-dependence of Odometer and Model Year
Using a correlation heat map we can see that both odometer and year are linearly related to selling price, but they are also related to each other.  To reduce colinearity while still preserving the data we made the following conversions.  
    - Convert year to age to capture the effect of an older vehicles
    - Divide Odometer by age to create miles_per_year and place a value on the relative wear on the vehicle 
A pearson R test confirmed that the new categories reduced the colinearity

### Exploring Vehicle condition
Because condition is a subjective measure I wanted to confirm that it was related to selling price.  Plotting the condition we see a relationship as expected, although it is a very noisy plot.  During modeling I'll try versions with and without condition to verify it's effectiveness.

### Takeaways from exploration:
- Distribution of the target variable, selling price, is a right-skewed normal distribution
- MMR is an excellent predictor of sales price. We'll try to beat it, then improve upon it
- Transmission, condition, make, model, color, and interior all drive the price, and we'll target-encode
- Odometer and year are co-linear. Converted to age_at_sale and miles_per_year to reduce colinearity

[[Back to top](#top)]

***

## <a name="model"></a>Modeling:

#### Modeling Results
SKLearn's Linear Regression model without using MMR as a feature:

    train:      $3985 RMSE : 79.3 R^2
    validate:   $3984 RMSE : 79.5 R^2
    
XGBoost Linear Regression model without using MMR as a feature:

    train:      $3002 RMSE : 88.2 R^2
    validate:   $3013 RMSE : 88.3 R^2
    
SKLearn's Linear Regression model with MMR as a feature:

    train:      $1480 RMSE : 97.1 R^2
    validate:   $1513 RMSE : 97.0 R^2
    
XGBoost Linear Regression model with MMR as a feature:

    train:      $1404 RMSE : 97.4 R^2
    validate:   $1442 RMSE : 97.3 R^2


- The standalone models without MMR as a feature were unable to beat the performance of the MMR model
- Using the MMR as a feature, we were able to reduce the MMR error by over $250

## Testing the Model

- XGBoost Model using MMR as a feature

#### Testing Dataset

| Model | RMSE on test | R2 score |
| ---- | ---- | ---- |
| XGBoost | $1406 |  97.4 |

[[Back to top](#top)]

***

## <a name="conclusion"></a>Conclusion and Next Steps:
Summary
The purpose of this project was utilize a database of used auto sales to construct a model that can predict sale price.
The first goal, to build a stand-alone model that beat the Manheim Market Report estimate, failed.

  -I was successfull at my second goal, to utilize the MMR estimate to improve upon their model.

  -I was able to utilize Skim, Pipeline, TargetEncoder, and XGBoost in my project, all of which simplified and improved my process and final product.

Drivers of Selling Price
Through initial testing of multiple models I was able to determine that the key drivers of a car's price were the make, age, model, and transmission type.

Expecations for Implementation
The consistant and outstanding performance on the test data indicates performance on new/unseen data would be high.

Recommendations
I recommend implemation of this model for business use at this time only if a stream of recent and reliable sales data can be obtained in order to keep the model current. Most of the sales in this dataset were from 2014-2015, so updated data to re-train the model would be required before using it to make purchase decisions.

Next Steps
With more time I would further tune the hyperparameters of the XGBoost model to try and obtain even better performance. Additionally, the VIN could be used to obtain more features about the vehicle such as engine size or technology packages. These could further improve model performance, but given that the current performance is more than adequate it may not be a wise investment of resources for what would likely be an incremental improvement.

[[Back to top](#top)]

*** 

## <a name="reproduce"></a>Steps to Reproduce:

  - Download the wrangle.py, explore.py, and final_notebook.ipynb
  - Download the car_prices.csv
  - Run the final_repot.ipynb notebook

[[Back to top](#top)]
