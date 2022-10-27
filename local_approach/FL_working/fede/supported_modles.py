from enum import Enum


class Supported_modles(str, Enum):
    logistic_regression = "LogisticRegression"
    SGD_classifier = "SGDClassifier"
    MLP_classifier = "MPL_classifier"
    rigde_classifier = "RidgeClassifier"
    gradient_boosting_classifier = "GradientBoostingClassifier"
