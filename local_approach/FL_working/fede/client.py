from time import sleep
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
)  # try to use different tools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import numpy as np
import random
from fed_transfer import Fed_Avg_Client
from supported_modles import Supported_modles
import socket, pickle
import sys

import argparse


class Client:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.accuracy = 0
        self.f1 = 0
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.feature_names = None
        self.token = None

        print(f'Creating {self.name}.')

    def load_data(self, path: str, csids=False) -> pd.DataFrame:
        """Load csv representation of pcap data, which have 44 feature and one label where 1 indicates malicious communicaiton and 0 benign."""

        if csids:
            hdrs = " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length2,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label"
            columns = hdrs.split(",")
            dataset = pd.read_csv(path, names=columns)
        else:
            dataset = pd.read_csv(path)

        return dataset

    def clean_dataset(self, df: pd.DataFrame):
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def preprocess_data(self, df: pd.DataFrame, ciids=False):
        """Preprocess and split data into X and y. Where X are features and y is the packet label. 
        String gata are One Hot encoded."""

        if ciids == True:
            df[" Label"] = df[" Label"].replace(
                [
                    "DoS slowloris",
                    "DoS Slowhttptest",
                    "DoS Hulk",
                    "DoS GoldenEye",
                    "Heartbleed",
                ],
                1,
            )
            df[" Label"] = df[" Label"].replace(["BENIGN"], 0)
            df = self.clean_dataset(df)
            X = df.iloc[:, :-1]
            feature_names = list(X.columns)
            X = X.to_numpy()
            y = df.iloc[:, -1]

        else:
            list_drop = ["id", "attack_cat"]
            df.drop(list_drop, axis=1, inplace=True)
            df_numeric = df.select_dtypes(include=[np.number])

            for feature in df_numeric.columns:
                if (
                    df_numeric[feature].max() > 10 * df_numeric[feature].median()
                    and df_numeric[feature].max() > 10
                ):
                    df[feature] = np.where(
                        df[feature] < df[feature].quantile(0.95),
                        df[feature],
                        df[feature].quantile(0.95),
                    )

            df_numeric = df.select_dtypes(include=[np.number])

            df_numeric = df.select_dtypes(include=[np.number])
            df_before = df_numeric.copy()
            for feature in df_numeric.columns:
                if df_numeric[feature].nunique() > 50:
                    if df_numeric[feature].min() == 0:
                        df[feature] = np.log(df[feature] + 1)
                    else:
                        df[feature] = np.log(df[feature])

            df_numeric = df.select_dtypes(include=[np.number])
            df_cat = df.select_dtypes(exclude=[np.number])

            for feature in df_cat.columns:
                if df_cat[feature].nunique() > 6:
                    df[feature] = np.where(
                        df[feature].isin(df[feature].value_counts().head().index),
                        df[feature],
                        f"{feature}_rest",
                    )

            df_cat = df.select_dtypes(exclude=[np.number])

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            ct = ColumnTransformer(
                transformers=[("encoder", OneHotEncoder(), [1, 2, 3])],
                remainder="passthrough",
            )
            feature_names = list(X.columns)
            X = np.array(ct.fit_transform(X))

            for label in list(df_cat["state"].value_counts().index)[::-1][1:]:
                feature_names.insert(0, label)

            for label in list(df_cat["service"].value_counts().index)[::-1][1:]:
                feature_names.insert(0, label)

            for label in list(df_cat["proto"].value_counts().index)[::-1][1:]:
                feature_names.insert(0, label)
            feature_names[5] = "service_not_determined"

        self.x = X
        self.y = y
        self.feature_names = feature_names

    def downsample_data(self, features):
        df = pd.DataFrame(self.x, columns=self.feature_names)
        self.x = df[features].to_numpy()
        self.feature_names = features

    def init_empty_model(self, model_name, model=None):
        if model_name == Supported_modles.SGD_classifier:
            self.model = SGDClassifier(
                n_jobs=-1,
                random_state=12,
                loss="log",
                learning_rate="optimal",
                eta0=0.15,
                verbose=0,
            )
        if model_name == Supported_modles.MLP_classifier:
            self.model = model
            self.model.intercepts_ = [
                np.zeros(40),
                np.zeros(25),
                np.zeros(5),
                np.zeros(1),
            ]
            self.model.coefs_ = [
                np.zeros((57, 40)),
                np.zeros((40, 25)),
                np.zeros((25, 5)),
                np.zeros((5, 1)),
            ]

    def split_data(self):
        self.x, self.x_test, self.y, self.y_test = train_test_split(
            self.x, self.y, test_size=0.33, random_state=random.randint(0, 10)
        )

    def prep_data(self):
        prep = StandardScaler()
        self.x = prep.fit_transform(self.x)
        self.x_test = prep.transform(self.x_test)

    def train_model(self, model_name):
        """ Train model on passed data. Curentlyy only LogReg is used. Function returns intercept and bias
        which later is being averaged with other model"""
        if model_name == Supported_modles.logistic_regression:
            clf = LogisticRegression(
                C=1.0,
                class_weight=None,
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                l1_ratio=None,
                max_iter=100,
                multi_class="auto",
                n_jobs=None,
                penalty="l2",
                random_state=13,
                solver="lbfgs",
                tol=0.0001,
                verbose=0,
                warm_start=False,
            ).fit(self.x, self.y)
        if model_name == Supported_modles.SGD_classifier:
            clf = SGDClassifier(
                random_state=32, loss="log", class_weight="balanced"
            ).fit(self.x, self.y)
        if model_name == Supported_modles.MLP_classifier:
            clf = MLPClassifier(
                solver="adam", alpha=1e-5, hidden_layer_sizes=(40, 25, 5)
            ).fit(self.x, self.y)
        if model_name == Supported_modles.rigde_classifier:
            clf = RidgeClassifier().fit(self.x, self.y)
        if model_name == Supported_modles.gradient_boosting_classifier:
            clf = GradientBoostingClassifier(
                n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
            ).fit(self.x, self.y)

        y_hat = clf.predict(self.x_test)

        self.model = clf
        self.accuracy = accuracy_score(self.y_test, y_hat)
        self.f1 = f1_score(self.y_test, y_hat)

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
        if y_test == None:
            y_hat = self.model.predict(self.x_test)
            return f1_score(self.y_test, y_hat)
        else:
            y_hat = self.model.predict(X_test)
            return f1_score(y_test, y_hat)

    def fed_avg_send_data(self, batch_size):
        X_train, X_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            shuffle=True,
            train_size=batch_size,
            stratify=self.y,
            random_state=random.randint(0, 10),
        )
        dataset_size = X_train.shape[0]
        sample_weights = compute_sample_weight("balanced", y=y_train)
        fed = Fed_Avg_Client(
            self.name, X_train, y_train, dataset_size, sample_weights, self.model
        )
        return fed

    def send_data_to_server(self, data):
        if self.token == None:
            print('Need to login first.')
            return
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", 5001))

            # Create an instance of ProcessData() to send to server.
            # Pickle the object and send it to the server
            data_string = pickle.dumps((self.token,data))
            s.send(data_string)
            print("Data Sent to Server")

    def wait_for_data(self):
        print('Waiting for connection')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # with conn:
        #     print(f"Connected by {addr}")
            s.connect(("localhost", 5001))
            data = b""
            while True:
                packet = s.recv(4096)
                if not packet:
                    break
                data += packet
            d = pickle.loads(data)
            return d
        
    def login(self):
        HOST = 'localhost'
        PORT = 5001
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            response = s.recv(2048)
            # Input UserName
            name = input(response.decode())	
            s.send(str.encode(name))
            response = s.recv(2048)
            # Input Password
            password = input(response.decode())	
            s.send(str.encode(password))
            ''' Response : Status of Connection :
                1 : Registeration successful 
                2 : Connection Successful
                3 : Login Failed
            '''
            # Receive response 
            response = s.recv(2048)
            response = response.decode()

            self.token = response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run client and get its ip and port.')
    parser.add_argument('--name', dest='name' , type=str, help='client name')
    # parser.add_argument('--port', dest='port' , type=int, help='port on which socket will run')
    # parser.add_argument('--address', dest='address', type=str, help='ip on which socket will run')
    parser.add_argument('--data', dest='data', type=str, help='ip on which socket will run')

    args = parser.parse_args()

    client = Client(args.name)
    
    dataset1 = client.load_data('../../../datasets/MachineLearningCSV/MachineLearningCVE/' + args.data, True)
    client.preprocess_data(dataset1, True)
    client.split_data()
    client.init_empty_model(Supported_modles.SGD_classifier)

    while True:
        cmd = input("stop, login, receive, score or send? ")
        if cmd == 'stop':
            break
        if cmd == 'login':
            client.login()
        if cmd == 'send':
            data = client.fed_avg_send_data(0.2)
            client.send_data_to_server(data)
        if cmd == 'receive':
            client.model = client.wait_for_data()
        if cmd == 'score':
            print(client.test_model_f1())
