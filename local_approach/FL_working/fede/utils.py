from fede.client import Client
from sklearn.utils import shuffle

# NODEs
def set_data(csids=False):
    client1 = Client("node1")

    if csids:
        client2 = Client("node2")
        client3 = Client("node3")
        client4 = Client("node4")
        client5 = Client("node5")

        dataset1 = client1.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedaa.csv', True)
        dataset2 = client2.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedab.csv', True)
        dataset3 = client3.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedac.csv', True)
        dataset4 = client4.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedad.csv', True)
        dataset5 = client5.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedae.csv', True)

        client1.preprocess_data(dataset1, True)
        client2.preprocess_data(dataset2, True)
        client3.preprocess_data(dataset3, True)
        client4.preprocess_data(dataset4, True)
        client5.preprocess_data(dataset5, True)

        client1.split_data()
        client2.split_data()
        client3.split_data()
        client4.split_data()
        client5.split_data()
        clients = [client1, client2, client3, client4, client5]

        
    else:
        dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")
        client2 = Client("node2")
        client3 = Client("node3")
        client4 = Client("node4")
        client5 = Client("node5")

        dataset = shuffle(dataset)


        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(dataset, csids)

        X = client1.x
        y = client1.y

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
    client1 = Client("node1", "0.0.0.0")
    dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")

    dataset = shuffle(dataset)

    client1.preprocess_data(dataset)

    return client1
