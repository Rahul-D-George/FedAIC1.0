from ConnectivityTests.Utils.Dataset import load_datasets
from ConnectivityTests.Client.CliUtils import *

net = Net()
trainloader, valloader, _ = load_datasets()
client = mk_client(net, trainloader, valloader)

# Connect to the Flower server
server_address = "apollo.doc.ic.ac.uk:6296"
fl.client.start_numpy_client(server_address=server_address, client=client)
