

from client import Client
import numpy as np

class Server(Client):
    def __init__(self, name, ip):
        super().__init__(name, ip)

    ###global zone
    def load_global_model(self, model):
        self.model = model
        self.global_model = True

    def update_global_model(self, applicable_models, round_weights)
        if self.global_model = False
            print(f'Not a global model.')
            return 0
        coefs = []
        intercepts = []
        for model in applicable_models:
            coefs.append(model.coef_)
            coefs.append(model.intercept_)

        self.model.coef_ = np.average(coefs, axis=0, weights=round_weights)
        self.model.intercept_ = np.average(intercepts, axis=0, weights=round_weights)

    def train_local_agent(self,X,y,model,epochs,class_weight):
        for _ in (0,epochs):
            model.partial_fit(X,y, classes=np.unique(y), sample_weight=class_weight)