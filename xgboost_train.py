import json
import os
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.utils import parallel_backend


def train_xgboost():
    # Define the XGBoost pipeline
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(objective='binary:logistic'))
    ])
    # Load the data from the JSON file
    with open('./ds_features.json', 'r') as f:
        data = json.load(f)

    # data_temp = []
    # for features, label in data:
    #     data_temp.append((np.array(features), label))
    X = np.array([np.array(x) for x, _ in data])
    y = np.array([int(y) for _, y in data])

    parameters = {
        'max_depth': [3, 5],
        'n_estimators': [10, 20, 30],
        'learning_rate': [0.01, 0.1, 0.25, 0.5, 1]
    }

    xgboost_classifier = XGBClassifier(objective='multi:softmax',
                                       num_class=3,
                                       eval_metric='logloss',
                                       missing=0,
                                       seed=123)

    kf_iterator = KFold(n_splits=3, shuffle=True, random_state=123)

    grd = GridSearchCV(xgboost_classifier,
                       param_grid=parameters,
                       n_jobs=multiprocessing.cpu_count() - 1,
                       cv=kf_iterator,
                       return_train_score=True)

    with parallel_backend('threading'):
        out = grd.fit(X=X, y=y)

    ds = pd.DataFrame(grd.cv_results_)
    print('hi')


if __name__ == '__main__':
    train_xgboost()
