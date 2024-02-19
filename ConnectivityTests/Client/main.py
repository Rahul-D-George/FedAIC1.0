from Utils.Dataset import load_datasets
from Client.CliUtils import *
import flwr as fl

net = Net()
trainloader, valloader, _ = load_datasets()
client = mk_client(net, trainloader, valloader)

# Connect to the Flower server
server_address = "192.168.1.106:8080" # PLACEHOLDER USED DURING TESTING
fl.client.start_numpy_client(server_address=server_address, client=client)
