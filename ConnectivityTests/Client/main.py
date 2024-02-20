from ConnectivityTests.Utils.Dataset import load_datasets
from ConnectivityTests.Client.CliUtils import *
import time

net = Net()
trainloader, valloader, _ = load_datasets()
client = mk_client(net, trainloader, valloader)

# Connect to the Flower server
server_address = "192.168.1.106:8080"

while True:
    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)
        break
    except Exception as e:
        print(f"Failed to connect to server: {e}. Retrying in 5 seconds...")
        time.sleep(5)
        continue
