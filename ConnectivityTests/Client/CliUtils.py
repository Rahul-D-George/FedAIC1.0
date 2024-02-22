# Various imports.
import flwr as fl
import sys
from pathlib import Path

# Add the parent directory to path.
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Custom imports.
from Utils.Net import *
from Utils.FedParams import *


# Custom client class.
# This class is used to interface with the flwr library. It is a subclass of fl.client.NumPyClient.
class FlowerClient(fl.client.NumPyClient):

    # Constructor.
    # We initialise the client with the neural network, and the training and validation data.
    # One extra field is added to keep track of the current round - in other words, the number of times the
    # model has been trained.
    # This is used when
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = 0

    # Method to get the parameters of the model.
    def get_parameters(self, config):
        return get_parameters(self.net)

    # Method to fit the model.
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        self.round += 1
        return get_parameters(self.net), len(self.trainloader), {}

    # Method to evaluate the model.
    # Here, we print the local and global loss and accuracy to compare the local and global models, to ensure
    # that the global model is improving.
    def evaluate(self, parameters, config):
        PLoss, PAccuracy = test(self.net, self.valloader)
        set_parameters(self.net, parameters)
        FLoss, FAccuracy = test(self.net, self.valloader)
        print(f"ROUND: {self.round} | LOCAL LOSS & ACCURACY : {float(PLoss)}, {float(PAccuracy) * 100}% "
              f"| GLOBAL LOSS & ACCURACY: {float(FLoss)}, {float(FAccuracy) * 100}%\n", flush=True)
        return float(FLoss), len(self.valloader), {"accuracy": float(FAccuracy)}


# Abstraction method to create a client.
def mk_client(net, trainloader, valloader) -> FlowerClient:
    return FlowerClient(net, trainloader, valloader)