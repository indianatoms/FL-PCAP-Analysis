from http import server
import socket
import os
import threading
import hashlib
import json

# Create Socket (TCP) Connection
class Server:
    def __init__(self):
        self.socket = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM) 
        self.hashtable = None

        host = '127.0.0.1'
        port = 1234
        ThreadCount = 0
        try:
            self.socket.bind((host, port))
        except socket.error as e:
            print(str(e))

        print('Waitiing for a Connection..')
        self.socket.listen(5)
        with open('ghost.txt', 'r') as f:
            self.hashtable = json.loads(f.read())

    # Function : For each client 
    def threaded_client(self, connection):
        connection.send(str.encode('ENTER USERNAME : ')) # Request Username
        name = connection.recv(2048)
        connection.send(str.encode('ENTER PASSWORD : ')) # Request Password
        password = connection.recv(2048)
        password = password.decode()
        name = name.decode()
        password=hashlib.sha256(str.encode(password)).hexdigest() # Password hash using SHA256
    # REGISTERATION PHASE   
    # If new user,  regiter in Hashtable Dictionary  
        if name not in self.hashtable:
            self.hashtable[name]=password
            connection.send(str.encode('Registeration Successful')) 
            print('Registered : ',name)
            print("{:<8} {:<20}".format('USER','PASSWORD'))
            for k, v in self.hashtable.items():
                label, num = k,v
                print("{:<8} {:<20}".format(label, num))
            print("-------------------------------------------")
            
        else:
    # If already existing user, check if the entered password is correct
            if(self.hashtable[name] == password):
                connection.send(str.encode('Connection Successful')) # Response Code for Connected Client 
                print('Connected : ',name)
            else:
                connection.send(str.encode('Login Failed')) # Response code for login failed
                print('Connection denied : ',name)
        while True:
            break
        connection.close()

if __name__ == "__main__":

    s1 = Server()
    ThreadCount = 1

    while True:
        Client, address = s1.socket.accept()
        # client_handler = threading.Thread(
        #     target=threaded_client,
        #     args=(Client,)  
        # )
        # client_handler.start()
        s1.threaded_client(Client)
        ThreadCount += 0
        if ThreadCount == 2:
            break
        print('Connection Request: ' + str(ThreadCount))

    s1.socket.close()