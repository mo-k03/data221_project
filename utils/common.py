import pandas as pd
from sklearn.model_selection import train_test_split

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

def ageAtSale(row):
    year = 2015
    if row['yr_renovated'] != 0:
        return year - row['yr_renovated']
    else:
        return year - row['yr_built']

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

    dataframe['age_at_sale'] = dataframe.apply(ageAtSale, axis=1)

    dataframe = dataframe.drop(['yr_built', 'yr_renovated', 'sqft_basement'], axis=1) #Turn the sqft of basements into has_basement, again, far more useful than its square feet.

    dataframe.to_csv('../../data/kc_house_data_cleaned.csv', index=False)

def trainTestSplit80_20(fileName):
    dataframe = pd.read_csv(fileName)

    features = dataframe.drop('price', axis=1) # defines features
    labels = dataframe['price'] # defines target label

    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.20, random_state=221) # ensures all models use the same random state and split

    return xTrain, xTest, yTrain, yTest
