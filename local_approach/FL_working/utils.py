from client import Client
from sklearn.utils import shuffle

#NODEs
def set_data(csids=False):
    client1 = Client("node1","0.0.0.0")

    if csids:
        dataset = client1.load_data("../../datasets/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")
    else:
        dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")

    client2 = Client("node2","0.0.0.0")
    client3 = Client("node3","0.0.0.0")
    client4 = Client("node4","0.0.0.0")
    client5 = Client("node5","0.0.0.0")

    dataset = shuffle(dataset)

    clients = [client1, client2, client3, client4, client5]

    client1.preprocess_data(dataset, csids)

    X = client1.x
    y = client1.y

    if csids:
        client1.x = X[:100000]
        client1.y = y[:100000]    

        client2.x = X[100000:200000]
        client2.y = y[100000:200000]

        client3.x = X[200000:300000]
        client3.y = y[200000:300000]

        client4.x = X[300000:400000]
        client4.y = y[300000:400000]

        client5.x = X[400000:]
        client5.y = y[400000:]
    else:
        client1.x = X[:20000]
        client1.y = y[:20000]    

        client2.x = X[20000:40000]
        client2.y = y[20000:40000]
        
        client3.x = X[40000:50000]
        client3.y = y[40000:50000]

        client4.x = X[50000:60000]
        client4.y = y[50000:60000]

        client5.x = X[60000:]
        client5.y = y[60000:]

    client2.feature_names = client1.feature_names
    client3.feature_names = client1.feature_names
    client4.feature_names = client1.feature_names
    client5.feature_names = client1.feature_names

    return clients

def centralized_data():
    client1 = Client("node1","0.0.0.0")
    dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")

    dataset = shuffle(dataset)

    client1.preprocess_data(dataset)

    return client1
