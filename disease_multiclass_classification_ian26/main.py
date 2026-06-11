import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.info()
test.info()

class_counts = pd.Series(train['Disease']).value_counts()

plt.figure()
class_counts.plot(kind = 'bar')
plt.xlabel('Clasa')
plt.ylabel("nr exemple")
plt.title("Distributia claselor")
#plt.show()

num_cols = ['Age','Symptom_Count']
gender_col = ['Gender']
symptom_col = 'Symptoms'

used_columns = num_cols + gender_col + [symptom_col]

X = train[used_columns]
X_test = test[used_columns]

le = LabelEncoder()
y = le.fit_transform(train['Disease'])

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

gender_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
])

symptom_pipeline = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(
        token_pattern=r'[^,]+',   
        ngram_range=(1,3),
        min_df=2,      
        max_features=5000,
        sublinear_tf=True        
    ))
])

preprocessor  = ColumnTransformer(transformers = [
    ('num', num_pipeline, num_cols),
    ('gender', gender_pipeline, gender_col),
    ('symptoms', symptom_pipeline, symptom_col)
])

X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)

svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC())
])

param_grid = {
    'classifier__C': [0.5, 1.0, 2.0, 5.0, 10.0]
}

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_tr, y_tr)

best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_val)

print("Best C:", grid.best_params_)
print("Accuracy:", accuracy_score(y_val, y_pred))

best_svm.fit(X,y)

y_test_pred_numeric = best_svm.predict(X_test)

y_test_pred_labels = le.inverse_transform(y_test_pred_numeric)

submission = pd.DataFrame({
    'Patient_ID': test['Patient_ID'],
    'Disease': y_test_pred_labels
})

submission.to_csv("submission_1_ian31st.csv", index=False)