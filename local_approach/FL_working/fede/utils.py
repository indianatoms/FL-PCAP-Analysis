from fede.client import Client
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# NODEs
def set_data(csids=False):
    client1 = Client("node1","localhost",50001)

    if csids:
        client2 = Client("node2","localhost",50001)
        client3 = Client("node3","localhost",50001)
        client4 = Client("node4","localhost",50001)
        client5 = Client("node5","localhost",50001)

    #Wednesdady dataset
        dataset = client1.load_data('datasets/Wednesday-workingHours.pcap_ISCX.csv', True)

        dataset = shuffle(dataset)

        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(dataset, csids)

        client1.prep_data()

        X = client1.x
        y = client1.y


        client1.x = X[:100000]
        client1.y = y[:100000]

        client2.x = X[100000:200000]
        client2.y = y[100000:200000]

        client3.x = X[200000:300000]
        client3.y = y[200000:300000]

        client4.x = X[300000:400000]
        client4.y = y[300000:400000]

        client5.x = X[400000:600000]
        client5.y = y[400000:600000]

        test_x = X[600000:]
        test_y = y[600000:]

        client2.feature_names = client1.feature_names
        client3.feature_names = client1.feature_names
        client4.feature_names = client1.feature_names
        client5.feature_names = client1.feature_names

        
    else:
        dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")

        client2 = Client("node2","localhost",50001)
        client3 = Client("node3","localhost",50001)
        client4 = Client("node4","localhost",50001)
        client5 = Client("node5","localhost",50001)

        dataset = shuffle(dataset)


        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(dataset, csids)
        client1.downsample_data(['sbytes','dbytes','sttl','dttl','spkts','dpkts'])


        X = client1.x
        y = client1.y

        client1.x = X[:20000]
        client1.y = y[:20000]

        client2.x = X[20000:30000]
        client2.y = y[20000:30000]

        client3.x = X[30000:40000]
        client3.y = y[30000:40000]

        client4.x = X[40000:50000]
        client4.y = y[40000:50000]

        client5.x = X[50000:60000]
        client5.y = y[50000:60000]

        test_x = X[60000:]
        test_y = y[60000:]

        prep = StandardScaler()
        test_x = prep.fit_transform(test_x)

        client2.feature_names = client1.feature_names
        client3.feature_names = client1.feature_names
        client4.feature_names = client1.feature_names
        client5.feature_names = client1.feature_names

    return clients, test_x, test_y


def centralized_data():
    client1 = Client("node1", "0.0.0.0",50001)
    dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")

    dataset = shuffle(dataset)

    client1.preprocess_data(dataset)

    return client1