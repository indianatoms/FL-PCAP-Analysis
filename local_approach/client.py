import pandas as pd
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class Client:
    def __init__(self, name, ip):
        self.name = name
        self.ip = ip

    def load_data(self, path: str) -> pd.DataFrame:
        """Load csv representation of pcap data, which have 44 feature and one label where 1 indicates malicious communicaiton and 0 benign."""

        dataset = pd.read_csv(path)
        return dataset

    def preprocess_data(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess and split data into X and y. Where X are features and y is the packet label. 
        String gata are One Hot encoded."""
        y = dataset.iloc[:,-1:]
        #one-hot-encode parameters
        proto = pd.get_dummies(dataset['proto'])
        state = pd.get_dummies(dataset['state'])
        service = pd.get_dummies(dataset['service'])

        #remove encoded parameters and add one hot
        for x in ['proto', 'state', 'service','label','attack_cat']:
            dataset = dataset.drop(x,axis = 1)
            
        for x in [proto, state, service]:
            dataset = dataset.join(x)
            
        X = dataset.iloc[:,:-2]

        return X, y
    
    def train_model(self, x: pd.DataFrame, y: pd.DataFrame)-> Tuple[np.ndarray, np.ndarray]:
        """ Train model on passed data. Curentlyy only LogReg is used. Function returns intercept and bias
        which later is being averaged with other model"""

        x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.33, random_state=42)
        clf1 = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                   l1_ratio=None, max_iter=100, multi_class='auto', n_jobs=None, penalty='l2', 
                   random_state=13, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False).fit(x_train1,y_train1)
        y_hat = clf1.predict(x_test1)
        accuracy_score(y_test1,y_hat)
        return clf1.coef_, clf1.intercept_