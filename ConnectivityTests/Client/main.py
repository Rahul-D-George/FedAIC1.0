import flwr as fl
from your_module_containing_FlowerClient import mk_client

from ConnectivityTests.Utils.Net import *
from ConnectivityTests.Client.CliUtils import *
from ConnectivityTests.Client.CliUtils import *


net = YourNeuralNetwork()
trainloader, valloader, _ = load_datasets()
client = mk_client(net, trainloader, valloader)

# Connect to the Flower server
server_address = "apollo.doc.ic.ac.uk:6296"
fl.client.start_numpy_client(server_address=server_address, client=client)
