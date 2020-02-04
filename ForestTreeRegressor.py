import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

data_link = "../data_sets/train.csv"

state_data = pd.read_csv(data_link)

# target variable
y = state_data.SalePrice

#list of features
features = ["LotArea", "YearBuilt",
            "1stFlrSF", "2ndFlrSF",
            "FullBath", "BedroomAbvGr",
            "TotRmsAbvGrd"]
X = state_data[features]

#separate data for train and test
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

#create a model
model = RandomForestRegressor(random_state=0, n_estimators=10, max_depth=10)

"""
    XGBoost : Extreme Gradient Boosting, is 
"""
XGBoost = XGBRegressor(n_estimator=700, learning_rate=0.5, n_jobs=10, objective='reg:squarederror')

#SVModel = svm.LinearSVC()

# fit model
model.fit(train_X, train_y)
XGBoost.fit(train_X, train_y, eval_set=[(val_X, val_y)], early_stopping_rounds=5, verbose=False)
#SVModel.fit(train_X, train_y)

# get_prediction
prediction_RFR = model.predict(val_X)
predictions_XGBR =  XGBoost.predict(val_X)
#predictions_SVMModel = SVModel.predict(val_X)
# calculat mean absolute error
print(mean_absolute_error(prediction_RFR, val_y))
print(mean_absolute_error(predictions_XGBR, val_y))
#print(mean_absolute_error(predictions_SVMModel, val_y))

# score
print(XGBoost.score(val_X, val_y))
"""
    When the prog is executed u will see a Future Warning ! It's weird, have no idea why that appear 
    when using xgboost !
    U may ignore the warning using the warning module. and ignore Future loadings, But i don't suggest or like that
"""

