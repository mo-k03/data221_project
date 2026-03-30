def randomForestModel():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    # Get directory of the current file, navigate up two levels in the folder (../..), converts the folders into something readable 
    # (i.e home/user/project), then finally, let python acknowledge it as a importable file.

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    import pandas as pd

    from utils.common import dataCleaning

    dataCleaning("data/kc_house_data.csv")

    dataframe = pd.read_csv("data/kc_house_data_cleaned.csv")

    features = dataframe.drop('price', axis=1)
    labels = dataframe['price']

    print("Splitting Dataset.....")

    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.20, random_state=221)

    print("Dataset successfully split.")

    model = RandomForestRegressor(
                    n_estimators=300, max_depth=20, random_state=221,
                    min_samples_leaf=2, min_samples_split=5, max_features=0.70
                    ) 

    #N Estimators = amount of decision trees in the forest. More trees = more consistent predictions.
    #Max Depth = How deep each tree is allowed to grow, each depth level adds another split.
    #Min Samples Leaf = Minimum amount of training samples for each leaf (where predictions are made). Prevents a tree from making very specific splits for one singular data point.
    #Min Samples Split = Minimum amount of samples needed before a node can be split. Prevents tiny splits from happening.
    #Max Features = Percentage of the features a tree should consider. (i.e 70% -> tree only considers 70% of features). This is the random in random forest.
    
    #In general, these tunes are here to soften the blow of an already overfitting-prone model.

    print("Fitting Model.....")

    model.fit(xTrain, yTrain)

    print("Model Fitted.")

    predictedTrain = model.predict(xTrain)
    predictedTest = model.predict(xTest)

    r2TrainingScore = r2_score(yTrain, predictedTrain)
    r2TestingScore = r2_score(yTest, predictedTest)
    mae = mean_absolute_error(yTest, predictedTest)
    mse = mean_squared_error(yTest, predictedTest)

    print(f"\n\nR^2 Training Score: {r2TrainingScore:.2f}\nR^2 Testing Score: {r2TestingScore:.2f}\nMean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}")

    # there is slight overfitting within the model, which is seen in the difference between the R^2 of the training, and testing score.
    # this amount of overfitting is generally normal within random forest models due to its complexity and amount of parameters,
    # i have attempted to tune the parameters to minimize overfitting and reduce the MAE and MSE as much as possible.

randomForestModel()

