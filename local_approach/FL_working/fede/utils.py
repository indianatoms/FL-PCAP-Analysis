from fede.client import Client
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# NODEs
def set_data(selected_model ,csids=False, downsample = False):
    client1 = Client("node1","0.0.0.0", 5001, selected_model)

    if csids:
        client2 = Client("node2","0.0.0.0", 5001, selected_model)
        client3 = Client("node3","0.0.0.0", 5001, selected_model)
        client4 = Client("node4","0.0.0.0", 5001, selected_model)
        client5 = Client("node5","0.0.0.0", 5001, selected_model)

#wed
        # dataset1 = client1.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedaa.csv', True)
        # dataset2 = client2.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedab.csv', True)
        # dataset3 = client3.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedac.csv', True)
        # dataset4 = client4.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedad.csv', True)
        # dataset5 = client5.load_data('../../datasets/MachineLearningCSV/MachineLearningCVE/newWedae.csv', True)


        # dataset1 = client1.load_data('datasets/Friday-DDosaa.csv', True)
        # dataset2 = client2.load_data('datasets/Friday-DDosab.csv', True)
        # dataset3 = client3.load_data('datasets/Friday-Morning.csv', True)
        # dataset4 = client4.load_data('datasets/Friday-PortScanaa.csv', True)
        # dataset5 = client5.load_data('datasets/Friday-PortScanab.csv', True)

        #'Wednesday-workingHours.pcap_ISCX.csv'
        #'Friday-WorkingHours-Morning.pcap_ISCX.csv'
        dataset = client1.load_data('datasets/Wednesday-workingHours.pcap_ISCX.csv', True)

        # dataset = shuffle(dataset)

        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(dataset, csids)
        if downsample:
            client1.downsample_data(['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets','Total Length of Fwd Packets'])


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

        # client1.x = X[:33000]
        # client1.y = y[:33000]

        # client2.x = X[33000:66000]
        # client2.y = y[33000:66000]

        # client3.x = X[66000:99000]
        # client3.y = y[66000:99000]

        # client4.x = X[99000:133000]
        # client4.y = y[99000:133000]

        # client5.x = X[133000:166000]
        # client5.y = y[133000:166000]

        # test_x = X[166000:]
        # test_y = y[166000:]

        client2.feature_names = client1.feature_names
        # client3.feature_names = client1.feature_names
        # client4.feature_names = client1.feature_names
        # client5.feature_names = client1.feature_names

        
    else:
        dataset = client1.load_data("../../datasets/UNSW_NB15_training-set.csv")

        client2 = Client("node2","localhost",50001, selected_model)
        client3 = Client("node3","localhost",50001, selected_model)
        client4 = Client("node4","localhost",50001, selected_model)
        client5 = Client("node5","localhost",50001, selected_model)

        dataset = shuffle(dataset)


        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(dataset, csids)
        if downsample:
            client1.downsample_data(['sbytes','dbytes','sttl','dttl','spkts','dpkts'])

        # client1.prep_data()

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