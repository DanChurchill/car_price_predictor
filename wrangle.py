import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_cars():
    '''
    function to read the dataset from local csv file
    '''
    return pd.read_csv('car_prices.csv', error_bad_lines=False, warn_bad_lines=False)

def carwash_1(df):
    df.transmission = df.transmission.fillna('unknown_transmission')
    df = df.dropna()

    # filter outliers from selling price
    q1, q3 = df.sellingprice.quantile([.25, .75]) # get quartiles
    iqr = q3 - q1   # calculate interquartile range
    upper_bound = q3 + 3 * iqr   # get upper bound
    lower_bound = q1 - 3 * iqr   # get lower bound
    df = df[(df['sellingprice'] > lower_bound) & (df['sellingprice'] < upper_bound)]


    df.drop_duplicates(subset="vin", keep=False, inplace=True) # remove duplicate VINs

    # consolidate body types into categories as best as possible
    df['body'].str.lower()
    df['body'] = np.where((df['body'].str.contains('van', case=False)), 'van', df.body)
    df['body'] = np.where((df['body'].str.contains('coupe', case=False)), 'coupe', df.body)
    df['body'] = np.where((df['body'].str.contains('sedan', case=False)), 'sedan', df.body)
    df['body'] = np.where((df['body'].str.contains('convertible', case=False)), 'convertible', df.body)
    df['body'] = np.where((df['body'].str.contains('wagon', case=False)), 'wagon', df.body)
    df['body'] = np.where((df['body'].str.contains('cab', case=False)), 'truck', df.body)
    df['body'] = np.where((df['body'].str.contains('koup', case=False)), 'coupe', df.body)
    df['body'] = np.where((df['body'].str.contains('crew', case=False)), 'truck', df.body)
    df['body'] = np.where((df['body'].str.contains('suv', case=False)), 'SUV', df.body)


    df['interior'] = np.where((df.interior == '—'), 'unknown_interior', df.interior) # change dashes to unknown_interior
    df['color'] = np.where((df.color == '—'), 'unknown_color', df.color)  # change dashes to 'unknown_color'

    df.odometer = df.odometer.astype(int) # change odometer to integer 
    df.state = df['state'].apply(lambda x: x.upper())  # uppercase state abbreviations
    df = df[df['make'].map(df['make'].value_counts()) > 100]  # remove rows with makes that have less than 100 samples
    df = df[df['state'].map(df['state'].value_counts()) > 750] # remove rows with states that have less than 750 samples
    df = df[df['color'].map(df['color'].value_counts()) > 1000] # remove colors with less than 1000 samples


    return df

def carwash_2(df):

    df['saleyear'] = df.saledate.str[11:16].astype(int)  # get sale year from sale date
    df['age_at_sale'] = df.saleyear - df.year  # create age at time of sale 
    df['age_at_sale'] = np.where((df['age_at_sale'] < 1), 1, df.age_at_sale) # for cars with age 0 or -1 change to 1

    df = df[df.odometer < 500000] # remove vehicles with over 500,000 miles


    # create miles per year column
    df['miles_per_year'] = (df.odometer / df.age_at_sale).astype(int)

    # drop columns not needed for modeling
    df.drop(columns=['seller', 'vin', 'saledate'], inplace=True)

    cols = ['make', 'model', 'trim', 'body', 'transmission', 'state',
            'color', 'interior']

    for col in cols:
        df[col] = df[col].astype(object)

    return df    

    

def split_cars(df):
    # separate into 80% train/validate and test data
    train_validate, test = train_test_split(df, test_size=.2, random_state=333)

    # further separate the train/validate data into train and validate
    train, validate = train_test_split(train_validate, 
                                    test_size=.25, 
                                    random_state=333)

    return train, validate, test