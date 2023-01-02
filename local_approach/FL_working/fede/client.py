import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import random
from fed_transfer import Fed_Avg_Client
from supported_modles import Supported_modles
from network import Net2nn
import socket, pickle
import torch
from torch import nn
from time import sleep
import time
import requests
from requests.auth import HTTPBasicAuth

import argparse


class Client:
    def __init__(self, name, server_address, server_port, model_name, conn):
        self.name = name
        self.server_address = server_address
        self.server_port = server_port
        self.model_name = model_name
        self.model = None
        self.accuracy = 0
        self.f1 = 0
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.feature_names = None
        self.token = None
        if conn == "socket":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


        print(f'Creating {self.name}.')

    def load_data(self, path: str, csids=False) -> pd.DataFrame:
        """Load csv representation of pcap data, which have 44 feature and one label where 1 indicates malicious communicaiton and 0 benign."""

        if csids:
            hdrs = " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length2,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label"
            columns = hdrs.split(",")
            columns = [x.strip(' ') for x in columns]
            dataset = pd.read_csv(path, names=columns)
        else:
            # hdrs = "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports,attack_cat,Label"
            # columns = hdrs.split(",")
            dataset = pd.read_csv(path)#, names=columns)

        return dataset

    def downsample(self, dataset):
        df_class_0 = dataset[dataset['Label'] == 0]
        df_class_1 = dataset[dataset['Label'] == 1]

        count_class_0 = df_class_0.shape[0]
        count_class_1 = df_class_1.shape[0]

        print(count_class_0)
        print(count_class_1)

        if count_class_0/(count_class_0 + count_class_1) < 0.2:
            print(f'DOWNSAMLING {self.name}')
            df_class_1_under = df_class_1.sample(count_class_0)
            dataset = pd.concat([df_class_1_under, df_class_0], axis=0)

        if count_class_1/(count_class_0 + count_class_1) < 0.2:
            print(f'DOWNSAMLING {self.name}')
            df_class_0_under = df_class_0.sample(count_class_1)
            dataset = pd.concat([df_class_0_under, df_class_1], axis=0)
        return dataset

    def clean_dataset(self, df: pd.DataFrame):
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def preprocess_data(self, df: pd.DataFrame, ciids=False, downsample = True):
        """Preprocess and split data into X and y. Where X are features and y is the packet label. 
        String gata are One Hot encoded."""

        if ciids == True:
            df["Label"] = df["Label"].replace(
                [
                    "DoS slowloris",
                    "DoS Slowhttptest",
                    "DoS Hulk",
                    "DoS GoldenEye",
                    "Heartbleed",
                    "DDoS",
                    "Bot",
                    "PortScan"
                ],
                1,
            )
            df["Label"] = df["Label"].replace(["BENIGN"], 0)

            df = self.clean_dataset(df)

            if downsample:
                df = self.downsample(df)
            X = df.iloc[:, :-1]
            feature_names = list(X.columns)
            X = X.to_numpy()
            y = df.iloc[:, -1]

        else:
            y = df.iloc[:, -1]
            list_drop = ["id", "attack_cat", "label"]
            df.drop(list_drop, axis=1, inplace=True)

            df_cat = df.select_dtypes(exclude=[np.number])

            for feature in df_cat.columns:
                if df_cat[feature].nunique() > 1:
                    df[feature] = np.where(
                        df[feature].isin(df[feature].value_counts().head().index),
                        df[feature],
                        f"{feature}_rest",
                    )
            
            for feature in df_cat.columns:
                one_hot = pd.get_dummies(df[feature])
                # Drop column B as it is now encoded
                df = df.drop(feature,axis = 1)
                # Join the encoded df
                df = df.join(one_hot)
            
            feature_names = list(df.columns)
            feature_names[45] = "service_not_determined"
            X = df.to_numpy()

        self.x = X
        self.y = y
        self.feature_names = feature_names
        

    def downsample_data(self, features):
        df = pd.DataFrame(self.x, columns=self.feature_names)
        self.x = df[features].to_numpy()
        self.feature_names = features
    
    def as_dataset(self):
        return pd.DataFrame(self.x, columns=self.feature_names)


    def init_empty_model(self, epochs = 20):
        if self.model_name == Supported_modles.SGD_classifier:
            self.model = SGDClassifier(
                loss="log",
                alpha=0.001,
                penalty='l1',
                max_iter=epochs
            )
        if self.model_name == Supported_modles.logistic_regression:
            self.model = LogisticRegression(
                C=100000,
                penalty="l2"
            )
        if self.model_name == Supported_modles.NN_classifier:
            self.model = Net2nn(self.x.shape[1])
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
            self.criterion = nn.CrossEntropyLoss()
            # self.model.fc1.weight.data.fill_(0.02)
            # self.model.fc2.weight.data.fill_(0.02)
            # self.model.fc3.weight.data.fill_(0.02)
            # self.model.fc1.bias.data.fill_(0.02)
            # self.model.fc2.bias.data.fill_(0.02)
            # self.model.fc3.bias.data.fill_(0.02)



    def split_data(self, test_size):
        return train_test_split(
            self.x, self.y, test_size=test_size, stratify=self.y, random_state=1
        )


    def prep_data(self):
        prep = StandardScaler()
        self.x = prep.fit_transform(self.x)

    def train_model(self, x=None, y=None, epochs=10):
        if self.model_name == Supported_modles.NN_classifier:
            if x is None:
                x = self.x
                y = self.y
            x_train = np.float32(x)  
            y_train = np.float32(y)

            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)

            self.train(x_train, y_train, epochs)
            return
        if x is None:
            self.model.fit(self.x, self.y)
        else:    
            self.model.fit(x, y)


    def partial_train_model(self, x=None, y=None):
        if x is None:
            x = self.x
            y = self.y
        self.model.partial_fit(x, y, classes=np.array([0, 1]))

    def train_local_agent(self, X, y, epochs, class_weight):
        if self.model_name == Supported_modles.SGD_classifier:
            for _ in range(0, epochs):
                self.model.partial_fit(
                    X, y, classes=np.unique(y), sample_weight=class_weight
                )
        if self.model_name == Supported_modles.NN_classifier:
                x_train = np.float32(X)  
                y_train = np.float32(y)

                x_train = torch.FloatTensor(x_train)
                y_train = torch.LongTensor(y_train)
                self.train(x_train, y_train, epochs) 

    def test_model_accuracy(self, y_test=None, X_test=None):
        if self.model is None:
            print("Model not trined yet.")
            return 0
        if y_test == None:
            y_hat = self.model.predict(self.x_test)
            return accuracy_score(self.y_test, y_hat)
        else:
            y_hat = self.model.predict(X_test)
            return accuracy_score(y_test, y_hat)

    def test_model_f1(self, y_test=None, X_test=None):
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

    def fed_avg_prepare_data(self, epochs):
        X_train, X_test, y_train, y_test = self.split_data(0.9)

        sample_weights = compute_sample_weight("balanced", y=y_train)

        self.train_local_agent(X_train,y_train,epochs,sample_weights)
        
        dataset_size = X_train.shape[0]

        fed = Fed_Avg_Client(
            self.name, dataset_size, self.model
        )
        return fed

    def send_data_to_server(self, data):
        if self.token == None:
            print('Need to login first.')
            return
        # Create an instance of ProcessData() to send to server.
        # Pickle the object and send it to the server
        data_string = pickle.dumps((self.token,data))
        self.socket.send(data_string)
        print("Data Sent to Server")

    def wait_for_data(self):
        data = b""
        # while True:
        packet = self.socket.recv(4096)
            # if not packet:
            #     break
        data += packet
        d = pickle.loads(data)
        return d

        
    def login_socket(self):
        HOST = self.server_address
        PORT = self.server_port

        self.socket.connect((HOST, PORT))
        response = self.socket.recv(2048)
        # Input UserName
        name = input(response.decode())	
        self.socket.send(str.encode(name))
        response = self.socket.recv(2048)
        # Input Password
        password = input(response.decode())	
        self.socket.send(str.encode(password))
        ''' Response : Status of Connection :
            1 : Registeration successful 
            2 : Connection Successful
            3 : Login Failed
        '''
        # Receive response 
        response = self.socket.recv(2048)
        response = response.decode()

        self.token = response


    def login_api(self):
        HOST = self.server_address
        PORT = self.server_port

        api_url = 'http://' + str(HOST) + ':' + str(PORT) + '/login'

        login = 'admin'#input('login:')
        password = 'admin'#input('password:')

        response = requests.get(api_url, auth=HTTPBasicAuth(login, password))
        self.token = response.json()['token']
        print('loged in succesfully!')

    def train(self, x, y, num_epochs):
        self.model.train()


        for _ in range(num_epochs):
            output = self.model(x)
            loss = self.criterion(output, y)
            self.optimizer.zero_grad()  #what is going on over here
            loss.backward()
            self.optimizer.step()
            # if _ % 10 == 0:
            #     print(loss)
    

    def load_global_model(self, model):
        #self.model = model
        if self.model_name == Supported_modles.NN_classifier:
            self.model.fc1.weight.data = model.fc1.weight.data 
            self.model.fc2.weight.data = model.fc2.weight.data 
            self.model.fc3.weight.data = model.fc3.weight.data 
            self.model.fc1.bias.data = model.fc1.bias.data
            self.model.fc2.bias.data = model.fc2.bias.data
            self.model.fc3.bias.data = model.fc3.bias.data 
        else:
            self.model = model

    def get_global_model(self):
        HOST = self.server_address
        PORT = self.server_port

        api_url = 'http://' + str(HOST) + ':' + str(PORT) + '/model/global'

        headers = {'x-access-token': self.token, 'Accept' : 'application/json', 'Content-Type' : 'application/json'}
        response = requests.get(api_url, headers = headers)

        resp = response.json()

        if self.model_name == Supported_modles.SGD_classifier or self.model_name == Supported_modles.logistic_regression:
            self.model.intercept_[0] = np.array(resp['models']['intercept'])
            self.model.coef_[0] = np.array(resp['models']['coefs'])
        
        if self.model_name == Supported_modles.NN_classifier:
            self.model.fc1.weight.data = torch.Tensor(np.array(resp['models']['coefs'][0]))
            self.model.fc2.weight.data = torch.Tensor(np.array(resp['models']['coefs'][1]))
            self.model.fc3.weight.data = torch.Tensor(np.array(resp['models']['coefs'][2]))
            self.model.fc1.bias.data = torch.Tensor(np.array(resp['models']['intercept'][0]))
            self.model.fc2.bias.data = torch.Tensor(np.array(resp['models']['intercept'][1]))
            self.model.fc3.bias.data = torch.Tensor(np.array(resp['models']['intercept'][2]))
        

    def send_local_model(self, fed):
        HOST = self.server_address
        PORT = self.server_port
        api_url = 'http://' + str(HOST) + ':' + str(PORT) + '/model'

        headers = {'x-access-token': self.token, 'Accept' : 'application/json', 'Content-Type' : 'application/json'}
        if self.model_name == Supported_modles.SGD_classifier or self.model_name == Supported_modles.logistic_regression:
            json_data = {"type":self.model_name.value,"model": { "intercept": fed.model.intercept_[0], "coefs":fed.model.coef_[0].tolist(), "dataset_size":fed.dataset_size} }
        elif self.model_name == Supported_modles.NN_classifier:  
            json_data = {"type":self.model_name.value,"model": { 
            "intercept": [
                fed.model.fc1.bias.clone().detach().numpy().tolist(),
                fed.model.fc2.bias.clone().detach().numpy().tolist(),
                fed.model.fc3.bias.clone().detach().numpy().tolist()
            ], 
            "coefs": [
                fed.model.fc1.weight.clone().detach().numpy().tolist(),
                fed.model.fc2.weight.clone().detach().numpy().tolist(),
                fed.model.fc3.weight.clone().detach().numpy().tolist(),   
            ], 
            "dataset_size":fed.dataset_size} }

        response = requests.post(api_url, headers = headers, json=json_data)
        print(response.status_code)
        print(response.text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run client and get its ip and port.')
    parser.add_argument('--name', dest='name' , type=str, help='client name')
    parser.add_argument('--port', dest='port' , type=int, help='port on which socket will run')
    parser.add_argument('--address', dest='address', type=str, help='ip on which socket will run')
    parser.add_argument('--data', dest='data', type=str, help='path to data')
    parser.add_argument('--conn', dest='conn', type=str, help='Use either socket or api')
    parser.add_argument('--model', dest='model', type=str, help='Specify model type: NN, LR, SGD')

    args = parser.parse_args()

    if args.model == "NN":
        supported_model = Supported_modles.NN_classifier
    elif args.model == "SGD":
        supported_model = Supported_modles.SGD_classifier
    elif args.model == "LR":
        supported_model = Supported_modles.logistic_regression
    else:
        raise ValueError("Only: NN, SGR or LR are support, specify one of those.")

    client = Client(args.name, args.address, args.port, supported_model, args.conn)
    
    dataset = client.load_data(args.data, True)
    client.preprocess_data(dataset, True)
    client.prep_data()
    client.x, client.x_test, client.y, client.y_test = client.split_data(0.1)
    client.init_empty_model()
    
    while True:
        cmd = input("stop, login, load, local, (reset) model, score or send? ")
        if cmd == 'stop':
            break
        if cmd == 'login':
            if args.conn == "api":
                client.login_api()
            if args.conn == "socket":
                client.login_socket()
        if cmd == 'load':
            # global_model = client.wait_for_data()
            # client.load_global_model(global_model)
            client.get_global_model()
        if cmd == 'send':
            if args.conn == "socket":
                for round in range (3):
                    print(f'Staring round: {round}')
                    print(f'Training model')
                    start_time = time.time()
                    data = client.fed_avg_prepare_data(epochs=10)
                    execution_time = time.time() - start_time
                    print(f'{execution_time} seconds')
                    sleep(5 - execution_time)
                    print(f'Sending Data')
                    client.send_data_to_server(data)
                    print(f'Wait for server')
                    global_model = client.wait_for_data()
                    client.load_global_model(global_model)
                    print(client.model.intercept_)
                    print(client.test_model_f1())
        if cmd == 'local':
            if args.conn == "api":
                data = client.fed_avg_prepare_data(epochs=10)
                client.send_local_model(data)
        if cmd == 'score':
            print(client.test_model_f1())
        if cmd == 'local':
            client.train_model(epochs = 50)
        if cmd == 'reset':
            client.init_empty_model()
        
