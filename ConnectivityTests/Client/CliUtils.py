import flwr as fl
from ConnectivityTests.Utils.Net import *
from ConnectivityTests.Utils.FedParams import *

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.acc_log = ""
        self.pre_agg_params = None

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        self.pre_agg_params = parameters
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        Ploss, Paccuracy = test(self.net, self.valloader)
        set_parameters(self.net, parameters)
        Floss, Faccuracy = test(self.net, self.valloader)
        round = config["current_round"]
        print(f"ROUND: {round} | LOCAL LOSS & ACCURACY : {float(Ploss)}, {float(Paccuracy) * 100} "
              f"| GLOBAL LOSS & ACCURACY: {float(Floss)}, {float(Faccuracy) * 100}%\n", flush=True)
        return float(Floss), len(self.valloader), {"accuracy": float(Faccuracy)}


def mk_client(net, trainloader, valloader) -> FlowerClient:
    return FlowerClient(net, trainloader, valloader)