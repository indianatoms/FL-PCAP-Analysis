import socket, pickle

print("Server is Listening.....")
HOST = 'localhost'
PORT = 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
files = []
num = 0

while True:
   
    s.listen(1)
    conn, addr = s.accept()
    print ('Connected by', addr)
    data = conn.recv(4096)
    data_variable = pickle.loads(data)
    conn.close()
    print (data_variable)
    files.append(data_variable)
    num += 1
    print (f'Data received from client{num}')
    if num == 5:
        break

print(files)