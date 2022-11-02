
from fedavg import Fedavg


        
fedavg = Fedavg("global", 0.05)

while True:
    Client, address = fedavg.socket.accept()
    fedavg.threaded_client(Client)
    if len(fedavg.connections) == 2:
        break

for c in fedavg.connections:
    print(f'{c[0]}:{c[1]}')

fedavg.socket.close()