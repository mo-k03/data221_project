import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.2, random_state=221)