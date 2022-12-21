from local_approach.FL_working.client import Client
from sklearn.utils import shuffle
import pandas as pd

# NODEs
def set_data(selected_model ,csids=False, downsample = False, data_shuffle = False):
    client1 = Client("node1","0.0.0.0", 5001, selected_model)
    client2 = Client("node2","0.0.0.0", 5001, selected_model)
    client3 = Client("node3","0.0.0.0", 5001, selected_model)
    client4 = Client("node4","0.0.0.0", 5001, selected_model)
    client5 = Client("node5","0.0.0.0", 5001, selected_model)

    if csids:
        dataset = client1.load_data('data/Wednesday-workingHours.pcap_ISCX.csv', True)

        if data_shuffle:
            dataset = shuffle(dataset)
            

        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(dataset, csids)
        if downsample:
            client1.downsample_data(['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets','Total Length of Fwd Packets'])


        client1.prep_data()

        X = client1.x
        y = client1.y

        test_x = X[:100000]
        test_y = y[:100000]


        client1.x = X[100000:200000]
        client1.y = y[100000:200000]

        client2.x = X[200000:300000]
        client2.y = y[200000:300000]

        client3.x = X[300000:400000]
        client3.y = y[300000:400000]

        client4.x = X[400000:500000]
        client4.y = y[400000:500000]

        client5.x = X[500000:600000]
        client5.y = y[500000:600000]

        client2.feature_names = client1.feature_names
        client3.feature_names = client1.feature_names
        client4.feature_names = client1.feature_names
        client5.feature_names = client1.feature_names

        
    else:
        dataset = client1.load_data("data/UNSW_NB15_train-set.csv")
        test_dataset = client2.load_data("data/UNSW_NB15_test-set.csv")
        df = pd.concat([dataset,test_dataset], ignore_index=True)


        if data_shuffle:
            df = shuffle(df)

        clients = [client1, client2, client3, client4, client5]

        client1.preprocess_data(df, csids)
        if downsample:
            client1.downsample_data(['sbytes','dbytes','sttl','dttl','spkts','dpkts'])
            client2.downsample_data(['sbytes','dbytes','sttl','dttl','spkts','dpkts'])

        client1.prep_data()


        X = client1.x
        y = client1.y

        client1.x = X[:34000]
        client1.y = y[:34000]

        client2.x = X[34000:68000]
        client2.y = y[34000:68000]

        client3.x = X[68000:102000]
        client3.y = y[68000:102000]

        client4.x = X[102000:142000]
        client4.y = y[102000:142000]

        client5.x = X[142000:180000]
        client5.y = y[142000:180000]

        test_x = X[180000:]
        test_y = y[180000:]

        client2.feature_names = client1.feature_names
        client3.feature_names = client1.feature_names
        client4.feature_names = client1.feature_names
        client5.feature_names = client1.feature_names

    return clients, test_x, test_y



def set_data_mock(selected_model ,csids=False, downsample = False):
    client1 = Client("node1","0.0.0.0", 5001, selected_model)
    client2 = Client("node2","0.0.0.0", 5001, selected_model)
    client3 = Client("node3","0.0.0.0", 5001, selected_model)

    if csids:
#wed
        # dataset1 = client1.load_data('datasets/Friday-DDosaa.csv', True)
        # dataset2 = client2.load_data('datasets/Friday-DDosab.csv', True)
        # dataset3 = client3.load_data('datasets/Friday-Morning.csv', True)
        # dataset4 = client4.load_data('datasets/Friday-PortScanaa.csv', True)
        # dataset5 = client5.load_data('datasets/Friday-PortScanab.csv', True)

        #'Wednesday-workingHours.pcap_ISCX.csv'
        #'Friday-WorkingHours-Morning.pcap_ISCX.csv'
        dataset1 = client1.load_data('data/mock_testsaa.csv', True)
        dataset2 = client1.load_data('data/mock_testsab.csv', True)
        dataset3 = client1.load_data('datasets/mock_testsac.csv', True)


        #dataset = shuffle(dataset)

        clients = [client1, client2, client3]

        client1.preprocess_data(dataset1, csids)
        client2.preprocess_data(dataset2, csids)
        client3.preprocess_data(dataset3, csids)

        client1.prep_data()
        client2.prep_data()
        client3.prep_data()

        client1.x, client1.x_test, client1.y, client1.y_test = client1.split_data(0.3)
        client2.x, client2.x_test, client2.y, client2.y_test = client2.split_data(0.3)
        client3.x, client3.x_test, client3.y, client3.y_test = client3.split_data(0.3)

    return clients



def centralized_data(selected_model ,csids=False, downsample = False, data_shuffle = False):
    client1 = Client("node1","0.0.0.0", 5001, selected_model)
    client2 = Client("node2","0.0.0.0", 5001, selected_model)
    if csids:
        dataset = client1.load_data('data/Wednesday-workingHours.pcap_ISCX.csv', True)
        if data_shuffle:
            dataset = shuffle(dataset)
        client1.preprocess_data(dataset, csids)
        client1.prep_data()

        X = client1.x[100000:]
        y = client1.y[100000:]

        test_x = client1.x[:100000]
        test_y = client1.y[:100000]

    else:
        dataset = client1.load_data("data/UNSW_NB15_train-set.csv")
        test_dataset = client2.load_data("data/UNSW_NB15_test-set.csv")
        df = pd.concat([test_dataset,dataset], ignore_index=True)

        if data_shuffle:
            df = shuffle(df)
        client1.preprocess_data(df, csids)
        client1.prep_data()

        X = client1.x[80000:]
        y = client1.y[80000:]

        test_x = client1.x[:80000]
        test_y = client1.y[:80000]
    return client1, X, y, test_x, test_y