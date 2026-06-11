import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train["dif_NT_MEV"] = train["NT"] - train["MEV"]
test["dif_NT_MEV"] = test["NT"] - test["MEV"]

train["medie_totala"] = (train["NT"] + train["MEV"] + train["MATE"] + train["MGIM"]) / 4
test["medie_totala"] = (test["NT"] + test["MEV"] + test["MATE"] + test["MGIM"]) / 4

#Subtask1 
dif_nt_mev = (test["NT"] - test["MEV"]).round(2)

sub1 = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": test["id"],
    "answer": dif_nt_mev
})

#Subtask2
rank_mev = test["MEV"].rank(ascending=False, method="min").astype(int)

sub2 = pd.DataFrame({
    "subtaskID": 2,
    "datapointID": test["id"],
    "answer": rank_mev
})

#Subtask3
num_cols = ['NT','MEV','MATE','MGIM']
cat_col = ['judet', 'gen']

X = train[num_cols + cat_col]
X_test = test[num_cols + cat_col]

y = train['status_admitere']


num_pipeline  = Pipeline(steps = [
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps =[
    ('encoder',  OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers = [
    ('num', num_pipeline, num_cols),
    ('cat',cat_pipeline, cat_col)
])

X_tr,X_val,y_tr,y_val = train_test_split(X,y, test_size = 0.2, stratify=y, random_state = 42)

#clsf = RandomForestClassifier(
#    n_estimators=1500,
#    max_depth=None,
#   min_samples_split=5,
#    min_samples_leaf=2,
#    random_state=42,
#    n_jobs=-1
#)

clsf = XGBClassifier(
    n_estimators=1200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('clf', clsf)
])

model.fit(X_tr,y_tr)
y_pred = model.predict(X_val)
acc = accuracy_score(y_val,y_pred)
print(f'The accuracy for the RandomForest Classifier model is {acc}')

model.fit(X,y)

test_pred = model.predict(X_test)

sub3 = pd.DataFrame({
    "subtaskID": 3,
    "datapointID": test["id"],
    "answer": test_pred
})

submission = pd.concat([sub1, sub2, sub3])
submission = submission.sort_values(["datapointID", "subtaskID"])

submission.to_csv("submission_4_feb17th.csv", index=False)
