# Various imports.
import flwr as fl
from flwr.common import Metrics, FitRes, Parameters, Scalar
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
import time
import torch
import numpy as np
from flwr.server.client_proxy import ClientProxy

# Add parent directory to path.
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Custom imports.
from Utils.Net import Net
from Utils.FedParams import get_parameters


# Custom method to evaluate the aggregated metrics.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


# Main method.
# To explain what I'm doing here, please see phase-level comments below.
while True:

    # We begin by initialising a new neural network - the only reason we do this is to get the initial parameters
    # which the server will use to initialise it's global model. These are converted to a format that flwr can use.
    net = Net()
    parameters = fl.common.ndarrays_to_parameters(get_parameters(net))

    # We then produce a date string - this will be used to save any models that we want to keep, as well as
    # the log file for this particular training run. It ensures that we can uniquely identify the files.
    date_string = time.strftime("%Y%m%d-%H%M%S")

    # This is the custom strategy that we use to train the model. It is a subclass of the FedAvg strategy, but
    # the only real difference is hat in the very last round, we save the model to a file.
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            # If we are in the last round, we save the model to a file.
            # This is done by converting the parameters back to a state_dict, and then saving that state_dict to a file.
            if aggregated_parameters is not None and server_round == 99:
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                net.load_state_dict(state_dict, strict=True)
                torch.save(net.state_dict(), f"final_params_{date_string}.pth")

            return aggregated_parameters, aggregated_metrics

    # To briefly explain, we have set our fraction_fit and fraction_evaluate to 1, to ensure that every client
    # is used for both training and evaluation. We have set the minimum number of clients to 2, to ensure that
    # we have enough clients to train and evaluate (in other words, we wait for both hospitals before we do anything).
    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters
    )

    # This logger simply saves the logs to a file, with the date string as the identifier.
    fl.common.logger.configure(identifier=f"training_logs_{date_string}", filename=f"log_{date_string}.txt")

    # We then start the server, using the strategy we have defined above.
    server_address = "apollo.doc.ic.ac.uk:6296"
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy
    )