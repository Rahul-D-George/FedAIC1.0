# Various imports.
import sys
from pathlib import Path
import torch
import time

# Add the parent directory to path.
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Custom imports.
from Utils.Dataset import load_datasets
from Client.CliUtils import *

# We begin by initialising the neural network that the client wants to train.
net = Net()

# We then load the training and validation data.
trainloader, valloader, _ = load_datasets()

# We then create a client object, which is a subclass of fl.client.NumPyClient.
client = mk_client(net, trainloader, valloader)

# We finally start the client, and connect it to the server.
# This should enable federated learning to begin.
server_address = "apollo.doc.ic.ac.uk:6296"
fl.client.start_client(server_address=server_address, client=client.to_client())

# Once the client has finished training, we save the final model to a file.
model_parameters = net.state_dict()
date_string = time.strftime("%Y-%m-%d_%H-%M-%S")
path = f"model_params_{date_string}.pth"
torch.save(model_parameters, path)