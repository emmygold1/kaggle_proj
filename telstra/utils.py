import numpy as np
from sklearn.cross_validation import train_test_split


def evaluate(res, y_true):
    nrows = res.shape[0]
    res = res + 1e-15
    probs = res[list(range(nrows)), y_true]
    return - np.sum(np.log(probs)) / nrows


def get_score(x, y, estimator, random_state=43):
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=.33, random_state=random_state)
    estimator.fit(x_train, y_train)
    score_train = estimator.score(x_train, y_train)
    score_valid = estimator.score(x_valid, y_valid)
    return score_train, score_valid


def get_evaluate(x, y, estimator, random_state=4332):
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=.33, random_state=random_state)
    estimator.fit(x_train, y_train)
    evaluate_train = evaluate(estimator.predict_proba(x_train), y_train)
    evaluate_valid = evaluate(estimator.predict_proba(x_valid), y_valid)
    return evaluate_train, evaluate_valid
