import json
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.utils import parallel_backend


def train_xgboost():
    """
    train xgboost model on our hand crafted features
    :return:
    """
    # Load the data from the JSON file
    with open('./ds_features.json', 'r') as f:
        data = json.load(f)

    X = np.array([np.array(x) for x, _ in data])
    y = np.array([int(y) for _, y in data])

    parameters = {
        # very low depth of trees and small number of estimators to prevent as much overfit as possible
        'max_depth': [3, 5],
        'n_estimators': [5, 10, 15, 20],
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

    df = pd.DataFrame(grd.cv_results_)
    df = df[['param_learning_rate', 'param_max_depth', 'param_n_estimators', 'mean_test_score', 'mean_train_score']]
    print(f"Best accuracy: {grd.best_score_}")
    print('complete results:')
    print(df)


if __name__ == '__main__':
    train_xgboost()
