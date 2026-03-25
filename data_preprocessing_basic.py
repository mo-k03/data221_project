import pandas as pd
# loads data from csv
data = pd.read_csv('kc_house_data.csv')

# removes unneeded columns
data = data.drop(columns=['id','date','zipcode'])

# adds 'age_at_sale' column that shows how long it had been since major building changes have been made
# uses 2015 as that is time of data creation, 'yr_renovated' if it exists, otherwise uses 'yr_built'
data['age_at_sale'] = data.apply(lambda row: 2015 - row['yr_renovated'] if row['yr_renovated'] > 0
                                    else 2015 - row['yr_built'], axis=1)

# removes redundant features that are accounted for in 'age_at_sale'
data.drop(columns=['yr_built','yr_renovated'])

# creates new file to be used for models
data.to_csv('kc_house_data_cleaned.csv', index=False)
