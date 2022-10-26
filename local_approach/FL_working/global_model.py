from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier #try to use different tools
import numpy as np
from supported_modles import Supported_modles
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier


class Global_model:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.accuracy = 0 
        self.F1 = 0
        

    def init_global_model(self, number_of_features, model_name, model):
        if model_name == Supported_modles.logistic_regression:
        # initialize global model
            clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                   l1_ratio=None, max_iter=100, multi_class='auto', n_jobs=None, penalty='l2', 
                   random_state=13, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
        if model_name == Supported_modles.SGD_classifier:
            clf = SGDClassifier(random_state=32, loss="log", class_weight="balanced")
        if model_name == Supported_modles.rigde_classifier:
            clf = RidgeClassifier()

        if model_name != Supported_modles.MLP_classifier:
            clf.intercept_ = np.zeros(1)
            clf.coef_ = np.zeros((1, number_of_features))
            if model_name != Supported_modles.rigde_classifier:
                clf.classes_ = np.array([0, 1])
            self.model = clf

        else:
            clf = model
            # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(40,25, 5), random_state=1)
            # clf.intercepts_ = [np.zeros(40), np.zeros(25), np.zeros(5), np.zeros(1)]
            # clf.coefs_ = [np.zeros((number_of_features,40)), np.zeros((40,25)), np.zeros((25,5)), np.zeros((5,1))]
            # clf.n_layers_ = 5
            # clf.out_activation_ = 'logistic'
            # clf.n_outputs_ = 1
            self.model = clf
            


    # update each agent model by current global model values
    def load_global_model(self, model):
        model.intercept_ = self.model.intercept_
        model.coef_ = self.model.coef_
        return model

    def update_global_model(self, applicable_clients, round_weights, model):
    # Average models parameters
        coefs = []
        intercept = []

        if model != Supported_modles.MLP_classifier:
            for client in applicable_clients:
                coefs.append(client.model.coef_)
                intercept.append(client.model.intercept_)
            
            self.model.coef_ = np.average(coefs, axis=0, weights=round_weights) # weight
            self.model.intercept_ = np.average(intercept, axis=0, weights=round_weights) # weight
        else:
            for client in applicable_clients:
                coefs.append(client.model.coefs_)
                intercept.append(client.model.intercepts_)
            
            self.model.coefs_ = np.average(coefs, axis=0, weights=round_weights) # weight
            self.model.intercepts_ = np.average(intercept, axis=0, weights=round_weights) # weight
        
    def f1_score(self, x_test,y_test):
        y_hat = self.model.predict(x_test)
        return f1_score(y_test,y_hat)