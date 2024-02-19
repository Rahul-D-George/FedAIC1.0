# Code to be ran on Imperial Apollo server.

import flwr as fl

fl.server.start_server(
    #server_address="apollo.doc.ic.ac.uk:6296",
    config=fl.server.ServerConfig(num_rounds=3)
)