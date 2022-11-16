from concurrent.futures import thread
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.neural_network import MLP_classifier
import socket, pickle, threading, hashlib, json, jwt, datetime, random
from supported_modles import Supported_modles
class Fedavg:
    def __init__(self, name, learning_rate):
        self.name = name
        self.model = None
        self.learning_rate = learning_rate
        self.accuracy = 0
        self.ip = '127.0.0.1'
        self.port = 5001
        self.clients = []
        self.hashtable = None
        self.clients = []
        self.secret = '5791628bb0b13ce0c676dfde280ba245'
        self.socket = None

        try:
            with open('ghost.json', 'r') as f:
                self.hashtable = json.loads(f.read())
        except FileNotFoundError:
            self.hashtable = None

        self.socket = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM) 
        ThreadCount = 0
        try:
            self.socket.bind((self.ip, self.port))
        except socket.error as e:
            print(str(e))

        print('Waitiing for a Connection..')
        self.socket.listen(5)


    def client_login(self, connection):
            connection.send(str.encode('ENTER USERNAME : ')) # Request Username
            name = connection.recv(2048)
            connection.send(str.encode('ENTER PASSWORD : ')) # Request Password
            password = connection.recv(2048)
            password = password.decode()
            name = name.decode()
            password=hashlib.sha256(str.encode(password)).hexdigest() # Password hash using SHA256
        # REGISTERATION PHASE   
        # If new user,  regiter in Hashtable Dictionary  
            if name not in self.hashtable:
                self.hashtable[name]=password
                connection.send(str.encode('Registeration Successful')) 
                print('Registered : ',name)
                print("{:<8} {:<20}".format('USER','PASSWORD'))
                for k, v in self.hashtable.items():
                    label, num = k,v
                    print("{:<8} {:<20}".format(label, num))
                print("-------------------------------------------")
                connection.close()
            else:
        # If already existing user, check if the entered password is correct
                if(self.hashtable[name] == password):
                    token = jwt.encode(
                    {
                        "name": name,
                        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
                    },
                        self.secret
                    )
                    print('Connected : ',token)
                    connection.send(token.encode())
                    connection.close()
                else:
                    connection.send(str.encode('False')) # Response code for login failed
                    print('Connection denied : ',name)
                    connection.close()

    def check_token(self, token):
            try:
                data = jwt.decode(token, self.secret, algorithms=['HS256'])
                current_user = data['name']
            except:
                return False
            print(current_user)
            return True


    def init_global_model(self, model_name, model, feature_numbers):
        if model_name == Supported_modles.SGD_classifier:
            self.model = SGDClassifier(
                loss="log",
                learning_rate="optimal",
                eta0=0.1,
                alpha=self.learning_rate
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

    def wait_for_data(self,connection):
        print('Waitiing for a Connection...')
        data = b""
        while True:
            packet = connection.recv(4096)
            if not packet:
                break
            data += packet
        d = pickle.loads(data)

        token = d[0]
        self.check_token(token)
        struct = d[1]

        self.clients.append(struct)
        connection.close()

    def send_request(self, connection, msg):
        print('Waitiing for a Connection...')
        data_string = pickle.dumps(msg)
        connection.send(data_string)
        connection.close()
        print("Data Sent to Server")

class ClientRefused(Exception):
    """Raised when one of the clinets does not agree to participate"""

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":

    NUMBER_OF_CLIENTS = 2
    fedavg = Fedavg("global", 0.05)
    ThreadCount = 0
    threads = []

    while True:
        Client, address = fedavg.socket.accept()
        client_handler = threading.Thread(
            target=fedavg.client_login,
            args=(Client,)  
        )
        client_handler.start()
        print('Connection Request: ' + str(ThreadCount))
        threads.append(client_handler)
        if len(threads) == NUMBER_OF_CLIENTS:
            break

    # Wait for all of them to finish
    for x in threads:
        x.join()
    
    fedavg.init_global_model(Supported_modles.SGD_classifier, None,78)

    selected_model = Supported_modles.SGD_classifier
    number_of_rounds = 3
    batch_size = 0.05
    epochs = 10

    for _ in range(number_of_rounds):

        print(f'Starting new round!')

        applicable_models = []
        applicable_name = []
        round_weights = []
        dataset_size = 0
        fedavg.clients = []
        threads = []
        
        while True:
            Client, address = fedavg.socket.accept()
            client_handler = threading.Thread(
                target=fedavg.wait_for_data,
                args=(Client,)  
            )
            client_handler.start()
            threads.append(client_handler)
            if len(threads) == NUMBER_OF_CLIENTS:
                break

        # Wait for all of them to finish
        for x in threads:
            x.join()
        
        applicable_clients = random.sample((fedavg.clients), random.randint(1, 2))


        for client in applicable_clients:
            print(client.name)
            round_weights.append(client.dataset_size)
            dataset_size += client.dataset_size
            print(round_weights)
            applicable_models.append(client.model)


        round_weights = np.array(round_weights) / dataset_size # calculate weight based on actual dataset size
        print(f'ROUNDS: {round_weights}')
        # round_weights = weights
        fedavg.update_global_model(applicable_models, round_weights, selected_model)
        print(fedavg.model.intercept_)

        ans = input("Send model to clients? ")

        if ans == "yes" or ans == "y":
            threads = []
            while True:
                Client, address = fedavg.socket.accept()
                client_handler = threading.Thread(
                    target=fedavg.send_request,
                    args=(Client,fedavg.model)  
                )
                client_handler.start()
                threads.append(client_handler)
                if len(threads) == NUMBER_OF_CLIENTS:
                    break
            for x in threads:
                    x.join()

    fedavg.socket.close()
        
