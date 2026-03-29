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

    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.20, random_state=221)

    model = RandomForestRegressor(
                    n_estimators=300, max_depth=20, random_state=221,
                    min_samples_leaf=2, min_samples_split=5, max_features=0.70
                    )
    model.fit(xTrain, yTrain)

    predictedTrain = model.predict(xTrain)
    predictedTest = model.predict(xTest)

    r2TrainingScore = r2_score(yTrain, predictedTrain)
    r2TestingScore = r2_score(yTest, predictedTest)
    mae = mean_absolute_error(yTest, predictedTest)
    mse = mean_squared_error(yTest, predictedTest)

    print(f"R^2 Training Score: {r2TrainingScore:.2f}\nR^2 Testing Score: {r2TestingScore:.2f}\nMean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}")

    # there is slight overfitting within the model, which is seen in the difference between the R^2 of the training, and testing score.
    # this amount of overfitting is generally normal within random forest models due to its complexity and amount of parameters,
    # i have attempted to tune the parameters to minimize overfitting and reduce the MAE and MSE as much as possible.

randomForestModel()