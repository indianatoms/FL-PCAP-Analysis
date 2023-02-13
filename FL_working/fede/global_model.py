from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
)  # try to use different tools
import numpy as np
from supported_modles import Supported_modles
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import torch


class GlobalModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.accuracy = 0
        self.f1 = 0

    def init_global_model(self, model):
        self.model = model

    # update each agent model by current global model values
    def load_global_model(self, model):
        model.intercept_ = self.model.intercept_
        model.coef_ = self.model.coef_
        return model

    def update_global_model(self, applicable_clients, round_weights, model):
        # Average models parameters
        coefs = []
        intercept = []

        if self.model_name == Supported_modles.logistic_regression:
            for client in applicable_clients:
                coefs.append(client.model.coef_)
                intercept.append(client.model.intercept_)

            self.model.coef_ = np.average(
                coefs, axis=0, weights=round_weights
            )  # weight
            self.model.intercept_ = np.average(
                intercept, axis=0, weights=round_weights
            )  # weight
        if self.model_name == Supported_modles.NN_classifier:
            if self.model_name == Supported_modles.NN_classifier:
                self.model = applicable_clients[0].model

            fc1_mean_weight = torch.zeros(
                size=applicable_clients[0].model.fc1.weight.shape
            )
            fc1_mean_bias = torch.zeros(size=applicable_clients[0].model.fc1.bias.shape)

            fc2_mean_weight = torch.zeros(
                size=applicable_clients[0].model.fc2.weight.shape
            )
            fc2_mean_bias = torch.zeros(size=applicable_clients[0].model.fc2.bias.shape)

            # fc3_mean_weight = torch.zeros(size=applicable_clients[0].model.fc3.weight.shape)
            # fc3_mean_bias = torch.zeros(size=applicable_clients[0].model.fc3.bias.shape)

            i = 0

            for client in applicable_clients:
                fc1_mean_weight += client.model.fc1.weight.data * round_weights[i]
                fc1_mean_bias += client.model.fc1.bias.data * round_weights[i]
                fc2_mean_weight += client.model.fc2.weight.data * round_weights[i]
                fc2_mean_bias += client.model.fc2.bias.data * round_weights[i]
                # fc3_mean_weight += client.model.fc3.weight.data * round_weights[i]
                # fc3_mean_bias += client.model.fc3.bias.data * round_weights[i]
                i += 1

            self.model.fc1.weight.data = fc1_mean_weight.data.clone()
            self.model.fc2.weight.data = fc2_mean_weight.data.clone()
            # self.model.fc3.weight.data = fc3_mean_weight.data.clone()
            self.model.fc1.bias.data = fc1_mean_bias.data.clone()
            self.model.fc2.bias.data = fc2_mean_bias.data.clone()
            # self.model.fc3.bias.data = fc3_mean_bias.data.clone()

    def f1_score(self, X_test, y_test):
        if self.model is None:
            print("Model not trined yet.")
            return 0
        if y_test is None:
            X_test = self.x_test
            y_test = self.y_test
        if self.model_name == Supported_modles.NN_classifier:
            test_x = np.float32(X_test)
            test_x = torch.FloatTensor(X_test)
            output = self.model(test_x)
            y_hat = output.argmax(dim=1, keepdim=True)
        else:
            y_hat = self.model.predict(X_test)
        return f1_score(y_test, y_hat, average="binary")

    def accuracy(self, x_test, y_test):
        y_hat = self.model.predict(x_test)
        return accuracy_score(y_test, y_hat)
