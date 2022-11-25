# Solution Architecture:

The final solution consists of two main classes - Client and Fedavg. They are both a part of a fede package.  

## Fedavg 
fedavg.py is acting as a server. It is responsible form obtaining the calculated models from the clients as well as it is performing the aggregation and finally it sends back the newly calculated model to the client. It is using the FedAvg algorithm, where the weight associated with parameters of the model are being adjusted accordingly to the amount of entries in the dataset. For the communication purposes, each FedAvg server has a socket that is being used to communicate with the clients. To achieve parallels when it comes to communication with the clients, it uses threading. Additionally, it is responsible for authenticating clients basing on username and password with is being compared with the private JSON file that store the data of currently enrolled users.

## Client 
client.py is a node in the FL scenario where it first needs to load the data and preprocess it. Next, it splits the dataset to have both a training and testing dataset. Thanks to which at the end it will be able in the end to compare the local model and the federate one. Client consists of 6 states:

1. Login - Login to the server and providing both username and password. If the authentication is successfully, it will obtain a token that later will be used for sending local models
2. Send - It triggers the FedAvg communication with the centralized client. It first prepares the Fed_Avg_Client that is used to transport the data needed in the 
3. Reset - It reset the model to initial state. It is useful to compare. 
4. Local - Train the model on the locally stored data
5. Load - **(Not implemented yet.)** It loads the saved model as a primary model.
6. Save - **(Not implemented yet)** It will save the current model. 
7. Score - Show the F1 score of the primary model on the local test data
8. Stop - Stop the program.

How to run a client: 
`python3 client.py --name node3 --address localhost --port 5001 --data ../datasets/newWedab.csv`

Client is taking 4 parameters.

1. --name - Specifies the name of the node. 
2. --address - Specifies the IP of the server to which it later will be connected.
3. --port -  Specifies the port of the server to which it later will be connected.
4. --data - Specifies the path to the dataset which will be used for the local training of the model.


How to run server: `python3 fedavg.py` - it does not take any parameters. In order to specify on which port, and IP address the server will be running, one needs to modify the line 17 and line 18 in the fedavg.py file. Additionally the number of clients and the number of round can be modified in lines 224 and 225.

The solution uses as well some helper classes such as:

**Fed_Avg_Client** - consists of three parameter - name, dataset size used for training and the model itself. It is used for communication to only send the relevant elements.
**Supported Models** - It is an enum which consists of all the models that are being supported by the program
**Network** - it is a file where the structure of the PyTorch based neural network is being stored. 
