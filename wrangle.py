import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_cars():
    '''
    function to read the dataset from local csv file
    '''
    return pd.read_csv('car_prices.csv', error_bad_lines=False, warn_bad_lines=False)

def clean_cars(df):
    df.transmission = df.transmission.fillna('unknown_transmission')
    df.transmission.value_counts()

    df = df.dropna()

    q1, q3 = df.sellingprice.quantile([.25, .75]) # get quartiles
    iqr = q3 - q1   # calculate interquartile range
    upper_bound = q3 + 3 * iqr   # get upper bound
    lower_bound = q1 - 3 * iqr   # get lower bound

    df = df[(df['sellingprice'] > lower_bound) & (df['sellingprice'] < upper_bound)]

    df = df[df.odometer < 500000] # remove vehicles with over 500,000 miles
    df = df[df.mmr < 100_000] # remove outlier MMR
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

    # value cleanup
    df['saleyear'] = df.saledate.str[11:16].astype(int)  # get sale year from sale date
    df['age_at_sale'] = df.saleyear - df.year  # create age at time of sale 
    df['age_at_sale'] = np.where((df['age_at_sale'] < 1), 1, df.age_at_sale) # for cars with age 0 or -1 change to 1
    df['interior'] = np.where((df.interior == '—'), 'unknown_interior', df.interior) # change dashes to unknown_interior
    df['color'] = np.where((df.color == '—'), 'unknown_color', df.color)  # change dashes to 'unknown_color'
    df.odometer = df.odometer.astype(int) # change odometer to integer 
    df.state = df['state'].apply(lambda x: x.upper())  # uppercase state abbreviations
    df = df[df['make'].map(df['make'].value_counts()) > 100]  # remove rows with makes that have less than 100 samples
    df = df[df['state'].map(df['state'].value_counts()) > 750] # remove rows with states that have less than 750 samples
    df = df[df['color'].map(df['color'].value_counts()) > 1000] # remove colors with less than 1000 samples

    # create miles per year column
    df['miles_per_year'] = (df.odometer / df.age_at_sale).astype(int)

    # create categorical groupings for make and drop original columns
    make_index = df.groupby(['make']).sellingprice.mean().to_frame().sort_values('sellingprice').index
    df['make_cat'] = ''
    df['make_cat'] = np.where((df.make.isin(make_index[:8])), 'cheap_make', df.make_cat)  
    df['make_cat'] = np.where((df.make.isin(make_index[8:19])), 'low_mid_make', df.make_cat)  
    df['make_cat'] = np.where((df.make.isin(make_index[19:29])), 'high_mid_make', df.make_cat)  
    df['make_cat'] = np.where((df.make.isin(make_index[29:])), 'luxury_make', df.make_cat)
    

    # create categorical groupings for state
    state_index = df.groupby(['state']).sellingprice.median().to_frame().sort_values('sellingprice').index
    df['state_cat'] = ''
    df['state_cat'] = np.where((df.state.isin(state_index[:7])), 'cheap_state', df.state_cat)  
    df['state_cat'] = np.where((df.state.isin(state_index[7:29])), 'mid_state', df.state_cat)  
    df['state_cat'] = np.where((df.state.isin(state_index[29:])), 'high_state', df.state_cat)

    # create categorical groupings for color
    color_index = df.groupby(['color']).sellingprice.median().to_frame().sort_values('sellingprice').index
    df['color_cat'] = ''
    df['color_cat'] = np.where((df.color.isin(color_index[:4])), 'low_color', df.color_cat)  
    df['color_cat'] = np.where((df.color.isin(color_index[4:10])), 'mid_color', df.color_cat)  
    df['color_cat'] = np.where((df.color.isin(color_index[10:])), 'high_color', df.color_cat) 

    # create categorical groupings for interior color
    interior_index = df.groupby(['interior']).sellingprice.median().to_frame().sort_values('sellingprice').index
    df['interior_cat'] = ''
    df['interior_cat'] = np.where((df.interior.isin(interior_index[:4])), 'low_interior', df.interior_cat)  
    df['interior_cat'] = np.where((df.interior.isin(interior_index[4:11])), 'mid_interior', df.interior_cat)  
    df['interior_cat'] = np.where((df.interior.isin(interior_index[11:])), 'high_interior', df.interior_cat)

    # create bins for trim
    trimlist = df.groupby(['trim']).mean().sort_values('sellingprice')
    trimlist['QuantileRank']= pd.qcut(trimlist['sellingprice'],
                             q = 10, labels = ['trim1', 'trim2', 'trim3', 'trim4', 'trim5',
                                               'trim6', 'trim7', 'trim8', 'trim9', 'trim10'])
    trimlist['trimcol'] = trimlist.index
    trim_dict = dict(zip(trimlist.trimcol, trimlist.QuantileRank))
    df['trim_cat']= df.trim.map(trim_dict)

    for cat in ['make_cat', 'state_cat', 'color_cat', 'interior_cat', 'body', 'transmission', 'trim_cat']:
        dummies = pd.get_dummies(df[cat], drop_first=True)
        df = pd.concat([df, dummies], axis=1)

    # drop columns not needed
    df.drop(columns=['interior', 'color', 'state', 'make', 'seller', 'trim', 'model', 'make_cat', 'state_cat',
                     'color_cat', 'transmission', 'vin', 'state', 'saledate', 'interior_cat', 'body', 'trim_cat'],
                      inplace=True)
    
    return df



def split_cars(df):
    # separate into 80% train/validate and test data
    train_validate, test = train_test_split(df, test_size=.2, random_state=333)

    # further separate the train/validate data into train and validate
    train, validate = train_test_split(train_validate, 
                                    test_size=.25, 
                                    random_state=333)

    return train, validate, test