import numpy as np 
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.info())
print(train.head())

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

test['month'] = test['date'].dt.month
test['year'] = test['date'].dt.year

y= train['gold close']
X = train.drop(columns = ['ID','date','gold close'])
test_enc = test.drop(columns = ['ID','date'])
 
regressor = GradientBoostingRegressor(
    loss =  'squared_error',
    n_estimators = 1200,
    max_depth = 7, 
    random_state = 42, 
    verbose = 1
)

model = Pipeline(steps = [
    ('impute', SimpleImputer(strategy = 'mean')),
    ('scale', StandardScaler()),
    ('regressor', regressor)
])

X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size= 0.2,random_state = 42)

model.fit(X_tr,y_tr)

y_pred = model.predict(X_val)

rmse = root_mean_squared_error(y_val,y_pred)
print(f'The rmse for gradient boosting {rmse}')

model.fit(X,y)

test_pred = model.predict(test_enc)

submission = pd.DataFrame({
    'ID' : test['ID'], 
    'gold close' : test_pred
})

print(submission)

submission.to_csv("submission_1_feb21st.csv")
