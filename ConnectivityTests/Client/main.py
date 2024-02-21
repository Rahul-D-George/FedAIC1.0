from ConnectivityTests.Utils.Dataset import load_datasets
from ConnectivityTests.Client.CliUtils import *
import torch
import time

net = Net()
trainloader, valloader, _ = load_datasets()
client = mk_client(net, trainloader, valloader)

# Connect to the Flower server
server_address = "apollo.doc.ic.ac.uk:6296"

fl.client.start_numpy_client(server_address=server_address, client=client)

model_parameters = net.state_dict()
date_string = time.strftime("%Y-%m-%d_%H-%M-%S")
path = f"model_params_{date_string}.pth"
torch.save(model_parameters, path)


# logs = client.acc_log
# with open(f"logs_{date_string}.txt", "w") as f:
#     f.write(logs)
#

# Attempted graceful reconnection
# while True:
#     try:
#         fl.client.start_numpy_client(server_address=server_address, client=client)
#         break
#     except Exception as e:
#         print(f"Failed to connect to server: {e}. Retrying in 5 seconds...")
#         time.sleep(5)
#         continue
