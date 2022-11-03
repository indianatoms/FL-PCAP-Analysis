import numpy as np
from time import sleep
from threading import Thread
from sklearn.linear_model import LogisticRegression, SGDClassifier
import timeout_decorator

# from sklearn.neural_network import MLP_classifier
import socket, pickle

from supported_modles import Supported_modles
import configparser

class Fedavg:
    def __init__(self, name, learning_rate):
        self.name = name
        self.model = None
        self.learning_rate = learning_rate
        self.accuracy = 0
        self.ip = "localhost"
        self.port = 5001
        self.clients = []

    def init_global_model(self, model_name, model, feature_numbers):
        if model_name == Supported_modles.SGD_classifier:
            self.model = SGDClassifier(
               random_state=32, loss="log", class_weight="balanced"
            )  # global
            # initialize global model
            self.model.intercept_ = np.zeros(1)
            self.model.coef_ = np.zeros((1, feature_numbers))
            self.model.classes_ = np.array([0, 1])
        if model_name == Supported_modles.MLP_classifier:
            clf = model
            self.model = clf

    def register_client(self, clients):
        self.clients = clients

    def update_global_model(self, applicable_models, round_weights, model_name):
        # Average models parameters
        coefs = []
        intercept = []
        if model_name == Supported_modles.SGD_classifier:
            for model in applicable_models:
                coefs.append(model.coef_)
                intercept.append(model.intercept_)
                # average and update FedAvg (aggregator model)
            self.model.coef_ = np.average(
                coefs, axis=0, weights=round_weights
            )  # weight
            self.model.intercept_ = np.average(
                intercept, axis=0, weights=round_weights
            )  # weight

        if model_name == Supported_modles.MLP_classifier:
            for model in applicable_models:
                coefs.append(model.coefs_)
                intercept.append(model.intercepts_)
            self.model.coefs_ = np.average(
                coefs, axis=0, weights=round_weights
            )  # weight
            self.model.intercepts_ = np.average(
                intercept, axis=0, weights=round_weights
            )  # weight

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
                model.partial_fit(
                    X, y, classes=np.unique(y), sample_weight=class_weight
                )
            if model_name == Supported_modles.MLP_classifier:
                model.partial_fit(X, y, classes=np.unique(y))

    # @timeout_decorator.timeout(10)
    def wait_for_data(self, number_of_clients):
        print("Server is Listening...")
 
        num = 0
        clients = []

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.ip, self.port))
            s.listen(2)

            while True:
                conn, addr = s.accept()
                print(f'Connected by {addr}')

                data = b""
                while True:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet

                d = pickle.loads(data)
                clients.append(d)

                num += 1
                print(f"Data received from client{num}")
                if num == number_of_clients:
                    break

        return clients

    def read_adresses(self, filepath):
        config = configparser.ConfigParser()
        config.read(filepath)
        config['workers']['ip'].splitlines()
        self.register_client(config['workers']['ip'].splitlines())

    def send_request(self, client_address, msg):
        x = client_address.split(":")
        HOST = x[0]
        PORT = int(x[1])
        # Create a socket connection.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            data_string = pickle.dumps(msg)
            s.send(data_string)

            s.close()
            print("Data Sent to Server")

class ClientRefused(Exception):
    """Raised when one of the clinets does not agree to participate"""
    pass


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    fedavg = Fedavg("global", 0.05)
    fedavg.init_global_model(Supported_modles.SGD_classifier, None,78)

    selected_model = Supported_modles.SGD_classifier
    number_of_rounds = 3
    batch_size = 0.05
    epochs = 10
    fedavg.read_adresses('server.conf')

    for c in fedavg.clients:
        print(c)
        fedavg.send_request(c,"Wanna connect?")

    answers = fedavg.wait_for_data(len(fedavg.clients))

    print(answers) #we can also ask for password

    for a in answers: ##Take only the guys that agreed + not timed out
        if a != 'yes' and a != 'y':
            raise ClientRefused
    
    sleep(5)
    for c in fedavg.clients:
        fedavg.send_request(c,"Start sending!")
            

    for _ in range(number_of_rounds):

        print(f'Starting new round!')

        applicable_clients = fedavg.wait_for_data(len(fedavg.clients))

        applicable_models = []
        applicable_name = []
        round_weights = []
        dataset_size = 0

        
        for client in applicable_clients:
            print(f'Client name: {client.name}')
            
            fedavg.load_global_model(client.model, selected_model) #load global model on the client model

            fedavg.train_local_agent(client.X_train, client.y_train, client.model, epochs, client.sample_weights, selected_model) #make partial fit on globsl model

            round_weights.append(client.X_train.shape[0])
            dataset_size += client.X_train.shape[0]
            print(round_weights)
            applicable_models.append(client.model)


        round_weights = np.array(round_weights) / dataset_size # calculate weight based on actual dataset size
        # round_weights = weights
        fedavg.update_global_model(applicable_models, round_weights, selected_model)
        print(fedavg.model.intercept_)

    sleep(5)
    for c in fedavg.clients:
        fedavg.send_request(c,fedavg.model)