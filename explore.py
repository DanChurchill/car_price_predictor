import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# plotting defaults
plt.rc('figure', figsize=(18, 9))
plt.style.use('seaborn-white')
plt.rc('font', size=16)

def correlation_test(df, col_1, col_2):
    '''
    Function performs linear correlation test between two columns of a dataframe
    and prints the results of the test.
    Accepts a dataframe, and two strings with the column names to be compared
    Prints results to console
    '''
    # perform test
    r , p = stats.pearsonr(df[col_1], df[col_2])
    
    # display Hypothesis
    print(f'H0: {col_1} is not linearly correlated to {col_2}')
    print(f'Ha: {col_1} is linearly correlated to {col_2}')
    print('')
    print ('Correlation coefficient is: {:.3f}'.format(r))
    
    # determine positive/negative correlation
    sign = ''      
    if r > 0:
          sign = 'positively'
    else:
          sign = 'negatively'
          
    # print results
    if p > .05:
        print('H0 is confirmed')
    else:
        print(f'H0 is rejected, {col_1} is {sign} correlated to {col_2}')


def transmission_plot(df):
    '''
    Modular function to display boxplot of selling price by vehicle transmission type
    Accepts a dataframe and displays a visualization
    '''
    ax = sns.boxplot(data=df, x='sellingprice', y='transmission')
    ax.set(title="Automatic Transmissions are more valuable than Manual",
           yticklabels=['Automatic', 'Unknown', 'Manual'],
           ylabel=None,
           xlabel='Selling Price')
    plt.grid(False)
    ax.xaxis.set_major_formatter('${x:1.0f}')
    plt.show()

def make_plot(df):
    '''
    Modular function to display a horizontal barplot of the top 5 and bottom 5 valued vehicle makes
    with a blank space between.  Accepts a dataframe and displays a visualization
    '''
    # get mean selling price grouped by Make
    values = pd.DataFrame(df.groupby('make').sellingprice.mean().sort_values(ascending=False))
    # get list of top/bottom 5 makes with a blank space in the middle
    y = values.head().index.tolist() + [' '] + values.tail().index.tolist()
    # get list of top/bottom 5 mean selling values with a blank space in the middle
    x = values.head().sellingprice.tolist() + [0] + values.tail().sellingprice.tolist()
    # combine into dataframe
    combo = pd.DataFrame()
    combo['Make'] = y
    combo['Selling Price'] = x
    combo['color'] = [1,1,1,1,1,0,0,0,0,0,0]
    # create plot
    ax = sns.barplot(data=combo, x='Selling Price', y='Make', hue='color')
    plt.title('Five Most and Least Valuable Auto Makes')
    plt.grid(False)
    ax.xaxis.set_major_formatter('${x:1.0f}')
    plt.legend([],[], frameon=False)
    plt.show()

def condition_plot(df):
    '''
    Modular function to display a plot of selling prices by vehicle condition rating
    accepts a dataframe and displays a visualization    
    '''
    x = sns.lineplot(data=df, x='sellingprice', y='condition')
    g = sns.lineplot(x=[0,50000], y=[2.6,4.1])
    x.set_xlabel('Selling Price')
    x.set_ylabel('Condition')
    x.xaxis.set_major_formatter('${x:1.0f}')
    plt.title('More expensive Cars have Higher Condition Ratings')
    plt.show()



def plot_dist(df):
    '''
    Modular function to display a distribution of vehicle selling prices, 
    along with the mean selling price
    accepts a dataframe and displays a visualization
    '''
    # Plot Distribution of target variable
    plt.figure(figsize=(24,12))
    sns.set(font_scale=2)
    plt.title('Distribution of Selling Prices')
    sns.histplot(data=df, x='sellingprice', stat='density')
    sns.kdeplot(data=df, x='sellingprice', color="red")
    plt.grid(False)
    plt.ylabel('Thousands of Sales')
    plt.xlabel('Selling Price')
    plt.ticklabel_format(style='plain', axis='both')
    plt.axvline(df.sellingprice.mean(), color='k', linestyle='dashed', linewidth=3)
    min_ylim, max_ylim = plt.ylim()
    plt.text(df.sellingprice.mean()*1.1, max_ylim*0.9, 'Average Price: ${:,}'.format(round(df.sellingprice.mean())))
    plt.show()

def mmr_plot(df):
    '''
    Modular function to display a scatter plot of MMR predicted prices 
    vs Actual Selling price
    accepts a dataframe and displays a visualization    
    '''
    ax = sns.scatterplot(df.sellingprice, y=df.mmr, label='MMR vs. Sales')
    b, a = np.polyfit(df.sellingprice, y=df.mmr, deg=1)
    plt.plot(df.sellingprice, df.sellingprice, color='r', lw=2.5, label='True Sales Price')
    ax.xaxis.set_major_formatter('${x:1.0f}')
    ax.yaxis.set_major_formatter('${x:1.0f}')
    plt.legend(frameon=True)
    plt.grid(False)
    plt.title('MMR predictions over Actual Selling Price')
    plt.xlabel('Selling Price')
    plt.ylabel('MMR Predicted Price')
    plt.show()

def heatmap(df):
    '''
    Modular function to display a heatmap of year, odometer, and sellingprice
    accepts a dataframe and displays a visualization
    '''
    # Identify columns and create correlation matrix
    cols = ['year', 'odometer', 'sellingprice']
    temp = df[cols].corr()

    # Plot the correlation
    sns.set(font_scale=2)
    sns.heatmap(temp, annot = True, mask= np.triu(temp), linewidth=.5, 
                cmap='Blues',annot_kws={"size": 20},linewidths=1, 
                linecolor='black', cbar=False)
    plt.grid(False)
    sns.set_theme(style='white')
    plt.title('Linear Correlation of Year, Odometer, and Selling Price', 
             fontdict= { 'fontsize': 24, 'fontweight':'bold'})
    plt.show()

def results_plot(df, preds):
    '''
    Modular function to plot model and MMR predications vs actual selling prices
    accepts a dataframe and a series and displays a visualization
    '''
    sns.set(font_scale=2)
    sns.set_style('dark')
    x = sns.scatterplot(df.sellingprice, df.mmr, label='MMR predictions')
    sns.scatterplot(df.sellingprice, preds, alpha=.3, label = 'XGBoost Predictions')
    sns.lineplot(df.sellingprice, df.sellingprice, alpha = 1, color='red', label= 'Actual Values')
    x.xaxis.set_major_formatter('${x:1.0f}')
    x.yaxis.set_major_formatter('${x:1.0f}')
    plt.xlabel('Selling Price')
    plt.ylabel('Predicted Price')
    plt.title('XGBoost predictions are closer to Actual Sale Price than MMR Predictions')
    plt.legend(frameon=True)
    plt.show()

def pred_eval(pipeline, X_in, y_in, X_out, y_out):
    '''
    Modular function to predict selling price, evaluate model performance,
    and display results.  Accepts a pre-fit pipeline containing a scaler and model,
    an in-sample dataframe of features, an in-sample column of actual values, and out-
    of-sample dataframe of features, and an out of sample column of actual values.
    Displays RMSE and R-squared values for in and out of sample performance.
    '''
    # predict train and validation set
    yhat_in = pipeline.predict(X_in)
    yhat_out = pipeline.predict(X_out)
    rmse_in = round(mean_squared_error(y_in, yhat_in, squared=False),2)
    rmse_out = round(mean_squared_error(y_out, yhat_out, squared=False),2)
    r2_in = round(r2_score(y_in, yhat_in),3)
    r2_out = round(r2_score(y_out, yhat_out),3)
    print(f'In-sample RMSE: {rmse_in}, In-sample r2: {r2_in}')
    print(f'Out-of-sample RMSE: {rmse_out}, Out-of-sample r2: {r2_out}')

def test_eval(pipeline, df):
    '''
    Modular function to test out of sample test data and display results
    accepts a fit pipeline containing a scaler and model, as well as a dataframe of test data.
    The function then makes predictions and displays RMSE and R-squared before calling the 
    results_plot function to display a visualization
    '''
    # predict train and validation set
    X = df.drop(columns= ['sellingprice','year'])
    y = df.sellingprice
    yhat = pipeline.predict(X)
    rmse = round(mean_squared_error(y, yhat, squared=False),2)
    r2 = round(r2_score(y, yhat),3)
    print(f'Test RMSE: {rmse}, Test r2: {r2}')

    results_plot(df, yhat)

# def plot_mmr(df):
#     plt.figure(figsize=(24,12))
#     sns.set(font_scale=1.5)
#     plt.title('MMR and Selling price')
#     sns.scatterplot(data=df, x='mmr', y='sellingprice')
#     plt.grid(False)
#     plt.show()

#     r , p = stats.pearsonr(df.mmr, df.sellingprice)
#     print ('Correlation coefficient is: {:.3f}'.format(r))
    
#     if p > .05:
#         print('Correlation is not confirmed')
#     else:
#         print('Correlation is confirmed')

# def corr_plot(df):
#     nums = ['odometer', 'mmr', 'age_at_sale', 'sellingprice', 'condition', 'miles_per_year']

#     # make correlation plot
#     df_corr = df[nums].corr()
#     plt.figure(figsize=(12,8))
#     sns.set(font_scale=1.5)
#     sns.heatmap(df_corr, cmap='Blues', annot = True, mask= np.triu(df_corr), linewidth=.5)
#     plt.show()


# def cat_plot(df):

#     cats = ['transmission', 'body', 'color', 'interior', 'state', 'make', 'make_cat',
#             'state', 'color_cat', 'interior_cat', 'trim_cat']
    
#     for cat in cats:
#         my_order = df.groupby(cat)["sellingprice"].median().sort_values().index
#         plt.figure(figsize=(12,8))
#         sns.set(font_scale=1)
#         sns.boxplot(data=df, x=cat, y='sellingprice', order=my_order)
#         plt.show()

# def compare_means(df,discrete_col,continuous_col):
#     group = df.groupby([discrete_col],as_index=False)[continuous_col].mean().reset_index(drop=True)
#     plt.figure(figsize=(10,5))
#     sns.barplot(x=group[discrete_col],y=group[continuous_col],palette='Reds')
#     plt.ylabel('mean ' + continuous_col)
#     plt.show()

def transmission_anova(df):
    '''
    Modular function to compare the mean selling price of vehicles by transmission type
    using an ANOVA analysis and print results to the console
    Accpets a dataframe and prints results
    '''
    auto = df[df.transmission == 'automatic']
    man = df[df.transmission == 'manual']
    unk = df[df.transmission == 'unknown_transmission']

    print('Selling price of automatics:            ', end='')
    print("${:.2f}".format(auto.sellingprice.mean()))
    print('Selling price of unknown transmissions: ', end='')
    print("${:.2f}".format(unk.sellingprice.mean()))
    print('Selling price of manuals:               ', end='')
    print("${:.2f}".format(man.sellingprice.mean()))

    print("")

    alpha = .05
    f, p = stats.f_oneway(auto.sellingprice, man.sellingprice, unk.sellingprice) 
    if p < alpha:
        print("We reject the Null Hypothesis, there is a significant difference in selling price between the different transmission types ")
    else:
        print("We confirm the Null Hypothesis")


