import numpy as np 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier

from fede.supported_modles import Supported_modles
from fede.fed_transfer import Fed_Avg_Client


class Fedavg:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.accuracy = 0
        self.ip = 'localhost'
        self.port = 5001

    
    def init_global_model(self, learnig_rate,model_name, model, feature_numbers):
        if model_name == Supported_modles.SGD_classifier:
            self.model = SGDClassifier(n_jobs=-1, random_state=12, loss="log", learning_rate='optimal', eta0=learnig_rate, verbose=0) # global
            # initialize global model
            self.model.intercept_ = np.zeros(1)
            self.model.coef_ = np.zeros((1, feature_numbers))
            self.model.classes_ = np.array([0, 1])
        if model_name == Supported_modles.MLP_classifier:
            clf = model
            self.model = clf

    def update_global_model(self, applicable_models, round_weights, model_name):
    # Average models parameters
        coefs = []
        intercept = []
        if model_name == Supported_modles.SGD_classifier:
            for model in applicable_models:
                coefs.append(model.coef_)
                intercept.append(model.intercept_)
                    # average and update FedAvg (aggregator model)
            self.model.coef_ = np.average(coefs, axis=0, weights=round_weights) # weight
            self.model.intercept_ = np.average(intercept, axis=0, weights=round_weights) # weight

        if model_name == Supported_modles.MLP_classifier:
            for model in applicable_models:
                coefs.append(model.coefs_)
                intercept.append(model.intercepts_)    
            self.model.coefs_ = np.average(coefs, axis=0, weights=round_weights) # weight
            self.model.intercepts_ = np.average(intercept, axis=0, weights=round_weights) # weight

    # update each agent model by current global model values
    def load_global_model(self, model, model_name):
        if model_name == Supported_modles.SGD_classifier:
            model.intercept_ = self.model.intercept_.copy()
            model.coef_ = self.model.coef_.copy()
        if model_name == Supported_modles.MLP_classifier:
            model.intercepts_ = self.model.intercepts_.copy()
            model.coefs_ = self.model.coefs_.copy()


    def train_local_agent(self, X, y, model, epochs, class_weight, model_name):
        for _ in range(0, epochs):
            if model_name == Supported_modles.SGD_classifier:
                model.partial_fit(X, y, classes=np.unique(y), sample_weight=class_weight)
            if model_name == Supported_modles.MLP_classifier:
                model.partial_fit(X, y, classes=np.unique(y))

    
    def wait_for_data(self, number_of_clients):
        import socket, pickle

        print("Server is Listening.....")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.ip, self.port))
        s.listen(2)
        conn, addr = s.accept()
        num = 0
        clients = []

        while True:
            data = b""
            while True:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            
            d = pickle.loads(data)
            clients.append(d)

            num += 1
            print (f'Data received from client{num}')
            if num == number_of_clients:
                 break

        # while True:
        #     s.listen(2)
        #     conn, addr = s.accept()
        #     data= conn.recv(1024).decode("ascii") 
        #     data = []
        #     while True:
        #         conn, addr = s.accept()
        #         packet = s.recv(4096)
        #         print ('Connected by', addr)
        #         if not packet: break
        #         data.append(packet)
        #     # data_arr = pickle.loads(b"".join(data))
        #     data_arr = pickle.loads(data)
        #     conn.close()
        #     print(type(data_arr))
        #     models.append(data_arr)
        #     num += 1
        #     print (f'Data received from client{num}')
        #     if num == number_of_clients:
        #         break

        return clients