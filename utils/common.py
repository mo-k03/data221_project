import pandas as pd 

def hasBasement(sqft_basement):
    if sqft_basement != 0:
        return 1
    else:
        return 0

def yearsSinceRenovation(yr_renovated, features):
    YEAR = 2015

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
    dataframe = pd.read_csv(FILE_NAME)

    dataframe = dataframe.drop(["id", "date", "zipcode"], axis=1) # Drop the redundant variables
    dataframe = removeOutliers(dataframe, "price") # Remove the price outliers from the dataframe using IQR

    dataframe['years_since_renovation'] = dataframe['yr_renovated'].apply(lambda value: yearsSinceRenovation(value, dataframe)) # Turn year renovated to the years since renovation,
    dataframe['has_basement'] = dataframe['sqft_basement'].apply(hasBasement)                            # which is far more useful for models than having "year renovated"

    dataframe = dataframe.drop(['yr_built', 'yr_renovated', 'sqft_basement'], axis=1) #Turn the sqft of basements into has_basement, again, far more useful than its square feet.

    dataframe.to_csv('data/kc_house_data_cleaned.csv', index=False)