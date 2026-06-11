import numpy as np 
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')

print(train.info())
print(test.info())

num_cols = ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Tutoring_Sessions',
            'Physical_Activity']
cat_cols = ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Motivation_Level','Internet_Access','Family_Income',
            'Teacher_Quality','School_Type','Peer_Influence','Learning_Disabilities','Parental_Education_Level','Distance_from_Home',
            'Gender']

X = train[num_cols+cat_cols]
X_test = test[num_cols+cat_cols]
y = train['Exam_Score']

xgb = XGBRegressor(
    n_estimators = 1200,
    max_depth = 7,
    learning_rate = 0.03,
    eval_metric = 'rmse',
    reg_alpha = 0.5,
    random_state = 42
)

num_pipeline = Pipeline(steps = [
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps = [
    ('encoder', OrdinalEncoder(
        handle_unknown  = 'use_encoded_value',
        unknown_value = -1
    ))
])

preprocessor = ColumnTransformer(transformers = [
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

model = Pipeline(steps = [\
    ('preprocessor', preprocessor),
    ('regressor', xgb)
])

X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size = 0.2,random_state = 42)

model.fit(X_tr,y_tr)

y_pred = model.predict(X_val)

print(f'The RMSE score for the xgb regressor is: {root_mean_squared_error(y_val,y_pred)}')

y_test = model.predict(X_test)

submission = pd.DataFrame({
    'SampleID' : test['SampleID'],
    'Exam_Score': y_test
})

submission.to_csv('submission_1_mar13th.csv')