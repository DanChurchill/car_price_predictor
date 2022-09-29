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
    r , p = stats.pearsonr(df[col_1], df[col_2])
    
    print(f'H0: {col_1} is not linearly correlated to {col_2}')
    print(f'Ha: {col_1} is linearly correlated to {col_2}')
    print('')
    print ('Correlation coefficient is: {:.3f}'.format(r))
    
    sign = ''      
    if r > 0:
          sign = 'positively'
    else:
          sign = 'negatively'
          
    
    if p > .05:
        print('H0 is confirmed')
    else:
        print(f'H0 is rejected, {col_1} is {sign} correlated to {col_2}')

def transmission_plot(df):
    ax = sns.boxplot(data=df, x='sellingprice', y='transmission')
    ax.set(title="Automatic Transmissions are more valuable than Manual",
           yticklabels=['Automatic', 'Unknown', 'Manual'],
           ylabel=None,
           xlabel='Selling Price')
    ax.xaxis.set_major_formatter('${x:1.0f}')
    plt.show()

def make_plot(df):
    values = pd.DataFrame(df.groupby('make').sellingprice.mean().sort_values(ascending=False))
    y = values.head().index.tolist() + [' '] + values.tail().index.tolist()
    x = values.head().sellingprice.tolist() + [0] + values.tail().sellingprice.tolist()
    combo = pd.DataFrame()
    combo['Make'] = y
    combo['Selling Price'] = x
    combo['color'] = [1,1,1,1,1,0,0,0,0,0,0]
    combo
    ax = sns.barplot(data=combo, x='Selling Price', y='Make', hue='color')
    plt.title('Five Most and Least Valuable Auto Makes')
    ax.xaxis.set_major_formatter('${x:1.0f}')

    plt.legend([],[], frameon=False)
    plt.show()

def condition_plot(df):
    x = sns.lineplot(data=df, x='sellingprice', y='condition')
    g = sns.lineplot(x=[0,50000], y=[2.6,4.1])
    x.set_xlabel('Selling Price')
    x.set_ylabel('Condition')
    x.xaxis.set_major_formatter('${x:1.0f}')
    plt.title('More expensive Cars have Higher Condition Ratings')
    plt.show()

def plot_dist(df):
    # Plot Distribution of target variable
    plt.figure(figsize=(24,12))
    sns.set(font_scale=2)
    plt.title('Distribution of Selling Prices')
    sns.histplot(data=df, x='sellingprice', stat='density')
    sns.kdeplot(data=df, x='sellingprice', color="red")
    plt.grid(False)
    plt.axvline(df.sellingprice.mean(), color='k', linestyle='dashed', linewidth=3)
    min_ylim, max_ylim = plt.ylim()
    plt.text(df.sellingprice.mean()*1.1, max_ylim*0.9, 'Average Price: ${:,}'.format(round(df.sellingprice.mean())))
    plt.show()

def mmr_plot(df):
    sns.scatterplot(df.sellingprice, y=df.mmr)
    b, a = np.polyfit(df.sellingprice, y=df.mmr, deg=1)
    xseq = np.linspace(0, 60000, num=100)
    plt.plot(xseq, a + b * xseq, color="r", lw=2.5)
    plt.show()

def results_plot(df, preds):
    sns.scatterplot(df.sellingprice, df.mmr, label='MMR predictions')
    sns.scatterplot(df.sellingprice, preds, alpha=.3, label = 'Model Predictions')
    sns.lineplot(df.sellingprice, df.sellingprice, alpha = 1, color='black', label= 'Actual Values')
    plt.legend()
    plt.show()

def pred_eval(pipeline, X_in, y_in, X_out, y_out):
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
    # predict train and validation set
    X = df.drop(columns= ['sellingprice','year'])
    y = df.sellingprice
    yhat = pipeline.predict(X)
    rmse = round(mean_squared_error(y, yhat, squared=False),2)
    r2 = round(r2_score(y, yhat),3)
    print(f'Test RMSE: {rmse}, Test r2: {r2}')

    results_plot(df, yhat)

def plot_mmr(df):
    plt.figure(figsize=(24,12))
    sns.set(font_scale=1.5)
    plt.title('MMR and Selling price')
    sns.scatterplot(data=df, x='mmr', y='sellingprice')
    plt.grid(False)
    plt.show()

    r , p = stats.pearsonr(df.mmr, df.sellingprice)
    print ('Correlation coefficient is: {:.3f}'.format(r))
    
    if p > .05:
        print('Correlation is not confirmed')
    else:
        print('Correlation is confirmed')

def corr_plot(df):
    nums = ['odometer', 'mmr', 'age_at_sale', 'sellingprice', 'condition', 'miles_per_year']

    # make correlation plot
    df_corr = df[nums].corr()
    plt.figure(figsize=(12,8))
    sns.set(font_scale=1.5)
    sns.heatmap(df_corr, cmap='Blues', annot = True, mask= np.triu(df_corr), linewidth=.5)
    plt.show()


def cat_plot(df):

    cats = ['transmission', 'body', 'color', 'interior', 'state', 'make', 'make_cat',
            'state', 'color_cat', 'interior_cat', 'trim_cat']
    
    for cat in cats:
        my_order = df.groupby(cat)["sellingprice"].median().sort_values().index
        plt.figure(figsize=(12,8))
        sns.set(font_scale=1)
        sns.boxplot(data=df, x=cat, y='sellingprice', order=my_order)
        plt.show()

def compare_means(df,discrete_col,continuous_col):
    group = df.groupby([discrete_col],as_index=False)[continuous_col].mean().reset_index(drop=True)
    plt.figure(figsize=(10,5))
    sns.barplot(x=group[discrete_col],y=group[continuous_col],palette='Reds')
    plt.ylabel('mean ' + continuous_col)
    plt.show()

def transmission_anova(df):
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


