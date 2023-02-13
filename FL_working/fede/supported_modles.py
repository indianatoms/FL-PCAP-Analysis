from enum import Enum


class Supported_modles(str, Enum):
    logistic_regression = "LogisticRegression"
    SGD_classifier = "SGDClassifier"
    NN_classifier = "NeuralNetworkClassifier"
