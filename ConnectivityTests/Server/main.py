# Code to be ran on Imperial Apollo server.

import flwr as fl
from flwr.common import Metrics

from ConnectivityTests.Utils.Net import *

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net()))
)

fl.server.start_server(
    #server_address="apollo.doc.ic.ac.uk:6296",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)