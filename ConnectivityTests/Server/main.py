# Code to be ran on Imperial Apollo server.

import flwr as fl
from flwr.common import Metrics, FitRes, Parameters, Scalar
from typing import List, Tuple, Dict, Optional, Union
from collections import OrderedDict
import os
import torch
import numpy as np
from flwr.server.client_proxy import ClientProxy

from ConnectivityTests.Utils.Net import Net
from ConnectivityTests.Utils.FedParams import get_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


while True:
    net = Net()
    c_dir = os.path.dirname(__file__)
    current_params = "model_params.pth"
    latest_params = os.path.join(c_dir, current_params)

    if not os.path.exists(latest_params):
        torch.save(net.state_dict(), latest_params)
        parameters = fl.common.ndarrays_to_parameters(get_parameters(net))
    else:
        print("Loading pre-trained model")
        state_dict = torch.load(latest_params)
        net.load_state_dict(state_dict)
        state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
        parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)


    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving aggregated_parameters...")

                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                net.load_state_dict(state_dict, strict=True)

                torch.save(net.state_dict(), f"model_params.pth")

            return aggregated_parameters, aggregated_metrics


    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters
    )

    fl.server.start_server(
        server_address="apollo.doc.ic.ac.uk:6296",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
