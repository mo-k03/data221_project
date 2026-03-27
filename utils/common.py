import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def hasBasement(sqft_basement):
    if sqft_basement != 0:
        return 1
    else:
        return 0

def yearsSinceRenovation(yr_renovated):
    meanYearBuilt = features['yr_built'].mean()
    
    if yr_renovated != 0:
        return int(YEAR - yr_renovated)
    else:
        return int(YEAR - meanYearBuilt)

def removeOutliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)

    IQR = Q3-Q1
    lower = Q1-(1.5*IQR)
    upper = Q3+(1.5*IQR)
    
    return dataframe[(dataframe[column] >= lower) & (dataframe[column] <= upper)]

def dataCleaning(FILE_NAME):
    YEAR = 2015

    dataframe = pd.read_csv(FILE_NAME)

    dataframe = dataframe.drop(["id", "date", "zipcode"], axis=1) # Drop the redundant variables
    dataframe = removeOutliers(dataframe, "price") # Remove the price outliers from the dataframe using IQR

    labels = dataframe['price']
    features = dataframe.drop('price',axis=1)

    features['years_since_renovation'] = features['yr_renovated'].apply(yearsSinceRenovation) # Turn year renovated to the years since renovation,
    features = features.drop(['yr_built', 'yr_renovated'],axis=1)                             # which is far more useful for models than having "year renovated"

    features['has_basement'] = features['sqft_basement'].apply(hasBasement) #Turn the sqft of basements into has_basement,
    features = features.drop('sqft_basement', axis=1)                        #again, far more useful than its square feet.

    dataframe.to_csv('data/kc_house_data_cleaned.csv', index=False)