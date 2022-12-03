import numpy as np
import socket, pickle, threading, hashlib, json, jwt, datetime, random
from supported_modles import Supported_modles
import torch
from sklearn.metrics import f1_score
import time
from token_expired_exception import TokenExpiredException

class Fedavg:
    def __init__(self, name, learning_rate, model_name):
        self.name = name
        self.model = None
        self.model_name = model_name
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
                time = data['exp']
            except:
                return False
            if time < int(time.time()):
                raise TokenExpiredException

            print(current_user)
            return True


    def init_global_model(self, model):
        self.model = model
        


    def register_client(self, clients):
        self.clients = clients

    def update_global_model(self, applicable_models, round_weights, model_name):
        # Average models parameters
        coefs = []
        intercept = []
        if self.model_name == Supported_modles.SGD_classifier:
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

        if self.model_name == Supported_modles.MLP_classifier:
            for model in applicable_models:
                coefs.append(model.coefs_)
                intercept.append(model.intercepts_)
            self.model.coefs_ = np.average(
                coefs, axis=0, weights=round_weights
            )  # weight
            self.model.intercepts_ = np.average(
                intercept, axis=0, weights=round_weights
            )  # weight

        if self.model_name == Supported_modles.NN_classifier:
            fc1_mean_weight = torch.zeros(size=applicable_models[0].fc1.weight.shape)
            fc1_mean_bias = torch.zeros(size=applicable_models[0].fc1.bias.shape)
    
            fc2_mean_weight = torch.zeros(size=applicable_models[0].fc2.weight.shape)
            fc2_mean_bias = torch.zeros(size=applicable_models[0].fc2.bias.shape)
            
            fc3_mean_weight = torch.zeros(size=applicable_models[0].fc3.weight.shape)
            fc3_mean_bias = torch.zeros(size=applicable_models[0].fc3.bias.shape)

            i = 0

            for model in applicable_models:
                fc1_mean_weight += model.fc1.weight.data.clone() * round_weights[i]
                fc1_mean_bias += model.fc1.bias.data.clone() * round_weights[i]
                fc2_mean_weight += model.fc2.weight.data.clone() * round_weights[i]
                fc2_mean_bias += model.fc2.bias.data.clone() * round_weights[i]
                fc3_mean_weight += model.fc3.weight.data.clone() * round_weights[i]
                fc3_mean_bias += model.fc3.bias.data.clone() * round_weights[i]
                i += 1
            
            model.fc1.weight.data = fc1_mean_weight.data.clone()
            model.fc2.weight.data = fc2_mean_weight.data.clone()
            model.fc3.weight.data = fc3_mean_weight.data.clone()
            model.fc1.bias.data = fc1_mean_bias.data.clone()
            model.fc2.bias.data = fc2_mean_bias.data.clone()
            model.fc3.bias.data = fc3_mean_bias.data.clone() 


    # update each agent model by current global model values
    def load_global_model(self, model, model_name):
        if model_name == Supported_modles.SGD_classifier:
            model.intercept_ = self.model.intercept_.copy()
            model.coef_ = self.model.coef_.copy()
        else:
            model = self.model

    def train_local_agent(self, X, y, model, epochs, class_weight):
        for _ in range(0, epochs):
            if self.model_name == Supported_modles.SGD_classifier:
                model.partial_fit(
                    X, y, classes=np.unique(y), sample_weight=class_weight
                )
            if self.model_name == Supported_modles.MLP_classifier:
                model.partial_fit(X, y, classes=np.unique(y))
            if self.model_name == Supported_modles.NN_classifier:
                x_train = np.float32(X)  
                y_train = np.float32(y)

                x_train = torch.FloatTensor(x_train)
                y_train = torch.LongTensor(y_train)
                self.train(x_train, y_train, 100)

    def test_model_f1(self, y_test=None, X_test=None):
        if self.model_name == Supported_modles.NN_classifier:
            test_x = np.float32(X_test)  
            test_x = torch.FloatTensor(X_test)
            output = self.model(test_x)
            prediction = output.argmax(dim=1, keepdim=True)
            return f1_score(prediction,y_test, average="binary")
        if self.model is None:
            print("Model not trined yet.")
            return 0
        if y_test is None:
            y_hat = self.model.predict(self.x_test)
            return f1_score(self.y_test, y_hat, average="binary")
        else:
            y_hat = self.model.predict(X_test)
            return f1_score(y_test, y_hat, average="binary")            
            

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



# Start Flower server for five rounds of federated learning
if __name__ == "__main__":

    NUMBER_OF_CLIENTS = 3
    NUMBER_OF_ROUNDS = 5
    
    selected_model = Supported_modles.NN_classifier
    fedavg = Fedavg("global", 0.1, selected_model)
    ThreadCount = 0
    threads = []

    #Timeout login porcess
    while True:
        Client, address = fedavg.socket.accept()
        client_handler = threading.Thread(
            target=fedavg.client_login,
            args=(Client,)  
        )
        client_handler.start()
        print(f'Connection Request: {len(threads)}')
        threads.append(client_handler)
        if len(threads) == NUMBER_OF_CLIENTS:
            break

    # Timeout all login processes
    for x in threads:
        x.join(30)
        if x.is_alive():
            x.terminate()
            x.join()

    
    epochs = 10
    max_score = 0
    optimal_model = None

    for round in range(NUMBER_OF_ROUNDS):
        print(f'Starting new round!')
        print(round, end=' ')

        applicable_models = []
        applicable_name = []
        round_weights = []
        threads = []
        dataset_size = 0
        
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
        
        applicable_clients = random.sample((fedavg.clients),2)# random.randint(1, 2))

        if round == 0:
            fedavg.model = applicable_clients[0].model

        for client in applicable_clients:
            print(f'.', end='')

            print(client.name)
            round_weights.append(client.dataset_size)
            dataset_size += client.dataset_size
            print(round_weights)
            applicable_models.append(client.model)


        round_weights = np.array(round_weights) / dataset_size
        fedavg.update_global_model(applicable_models, round_weights, selected_model)

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
        
