import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data_link = "./data_sets/train.csv"

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
model = RandomForestRegressor(random_state=0)

# fit model
model.fit(train_X, train_y)

# get_prediction
predictions = model.predict(val_X)

# calculat mean absolute error
mea = mean_absolute_error(predictions, val_y)

print(mea)

