# Code to be ran on Imperial Apollo server.

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple

from Utils.Net import *
from Utils.FedParams import get_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.5,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net()))
)

fl.server.start_server(
    # server_address="apollo.doc.ic.ac.uk:6296", EXCLUDED FOR NOW BUT VERY IMPORTANT
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)