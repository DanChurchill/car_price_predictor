import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

# plotting defaults
plt.rc('figure', figsize=(18, 9))
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

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


