import numpy as np 
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

#Subtask 1 

task1_answer = len(
    test[(test["City A"] == "Barlad") & (test["Weather"] == "Fog")]
)

#Subtask 2 

cat_cols = ['City A','City B','Weather']
num_cols = ['Distance','Time of Day','Traffic','Road Quality','Driver Experience']

from catboost import CatBoostRegressor

features = cat_cols + num_cols
target = "deliver_time"

X = train[features]
y = train[target]

X_test = test[features]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    cat_features=cat_cols,
    bagging_temperature=0.3,
    random_strength=1.0,
    early_stopping_rounds=100,
    verbose=False
)

model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val)
)

val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print("Validation MAE:", mae)

model.fit(X, y, cat_features=cat_cols, verbose=False)

test_preds = model.predict(X_test)
test_preds = np.round(test_preds, 3)

n_test = len(test)

submission = pd.DataFrame({
    "subtaskID": ["1"] + ["2"] * n_test,
    "datapointID": [1] + test["ID"].tolist(),
    "answer": [task1_answer] + test_preds.tolist()
})

submission.to_csv("output_2_cargo.csv", index=False)
