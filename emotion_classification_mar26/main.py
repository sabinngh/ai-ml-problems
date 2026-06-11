import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train['text'].astype(str)
y = train['label'].astype(str)
X_test = test['text'].astype(str)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

word_tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    min_df=5,
    max_df=0.9,
    max_features=50000,
    sublinear_tf=True,
    lowercase=True,
    strip_accents='unicode'
)

char_tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    min_df=2,
    sublinear_tf=True,
    lowercase=True
)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(
    C=7,
    max_iter=10000,
    class_weight="balanced"
)

model = Pipeline(steps=[
    ('features', FeatureUnion([
        ('word', word_tfidf),
        ('char', char_tfidf)
    ])),
    ('clf', classifier)
])

model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)

f1 = f1_score(y_val, y_pred, average='macro')

print(f'The f1 score for the linear svc and tfidf is {f1}')

model.fit(X,y)
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'SampleID' : test['SampleID'],
    'label' : test_pred
})



#submission.to_csv('submission_3_mar5th.csv')