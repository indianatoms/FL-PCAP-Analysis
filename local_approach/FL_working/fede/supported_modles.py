from enum import Enum


class Supported_modles(str, Enum):
    logistic_regression = "LogisticRegression"
    SGD_classifier = "SGDClassifier"
    MLP_classifier = "MLP_classifier"
    gradient_boosting_classifier = "GradientBoostingClassifier"
    NN_classifier = "NeuralNetworkClassifier"
