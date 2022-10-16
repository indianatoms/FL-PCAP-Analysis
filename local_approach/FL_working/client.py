import pandas as pd
from typing import Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier #try to use different tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
from supported_modles import Supported_modles

class Client:
    def __init__(self, name, ip):
        self.name = name
        self.ip = ip
        self.model = None
        self.accuracy = 0 
        self.F1 = 0
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.feature_names = None


    def load_data(self, path: str) -> pd.DataFrame:
        """Load csv representation of pcap data, which have 44 feature and one label where 1 indicates malicious communicaiton and 0 benign."""

        dataset = pd.read_csv(path)
        return dataset

    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess and split data into X and y. Where X are features and y is the packet label. 
        String gata are One Hot encoded."""

        list_drop = ['id','attack_cat']
        df.drop(list_drop,axis=1,inplace=True)
        df_numeric = df.select_dtypes(include=[np.number])

        for feature in df_numeric.columns:
            if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
                df[feature] = np.where(df[feature]<df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

        df_numeric = df.select_dtypes(include=[np.number])

        df_numeric = df.select_dtypes(include=[np.number])
        df_before = df_numeric.copy()
        for feature in df_numeric.columns:
            if df_numeric[feature].nunique()>50:
                if df_numeric[feature].min()==0:
                    df[feature] = np.log(df[feature]+1)
                else:
                    df[feature] = np.log(df[feature])

        df_numeric = df.select_dtypes(include=[np.number])
        df_cat = df.select_dtypes(exclude=[np.number])

        for feature in df_cat.columns:
            if df_cat[feature].nunique()>6:
                df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], f'{feature}_rest')

        df_cat = df.select_dtypes(exclude=[np.number])

        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3])], remainder='passthrough')
        feature_names = list(X.columns)
        X = np.array(ct.fit_transform(X))

        for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
            feature_names.insert(0,label)
            
        for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
            feature_names.insert(0,label)
            
        for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
            feature_names.insert(0,label)
        feature_names[5] = "service_not_determined"

        self.x = X
        self.y = y
        self.feature_names = feature_names


    def split_data(self):
        self.x, self.x_test, self.y, self.y_test = train_test_split(self.x, self.y, test_size=0.33, random_state=random.randint(0,10))

    
    def train_model(self, model_name):
        """ Train model on passed data. Curentlyy only LogReg is used. Function returns intercept and bias
        which later is being averaged with other model"""
        if model_name == Supported_modles.logistic_regression:
            clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                   l1_ratio=None, max_iter=100, multi_class='auto', n_jobs=None, penalty='l2', 
                   random_state=13, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False).fit(self.x,self.y)
        if model_name == Supported_modles.SGD_classifier:
            clf = SGDClassifier(random_state=32, loss="log", class_weight="balanced").fit(self.x,self.y)
        if model_name == Supported_modles.MLP_classifier:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40,25, 5), random_state=1).fit(self.x, self.y)

        y_hat = clf.predict(self.x_test)

        self.model = clf
        self.accuracy = accuracy_score(self.y_test,y_hat)
        self.F1 = f1_score(self.y_test,y_hat)

    def test_model_accuracy(self,y_test = None,X_test = None):
        if self.model is None:
            print ('Model not trined yet.')
            return 0
        if y_test == None:
            y_hat = self.model.predict(self.x_test)
            return accuracy_score(self.y_test,y_hat)
        else:    
            y_hat = self.model.predict(X_test)
            return accuracy_score(y_test,y_hat)

    def test_model_f1(self,y_test = None,X_test = None):
        if self.model is None:
            print ('Model not trined yet.')
            return 0
        if y_test == None:
            y_hat = self.model.predict(self.x_test)
            return f1_score(self.y_test,y_hat)
        else:
            y_hat = self.model.predict(X_test)
            return f1_score(y_test,y_hat)
