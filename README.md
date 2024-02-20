# Fed-AIC1.0
This is a newer version of the AI-Clinician<sup>[\[1\]][4]</sup> from the referenced paper. The old version implements reinforcement learning through a tabular method of maintaining Q-values; we instead use a DQN here.
Further, we make use of federated learning in order to make the process of training more efficient and private.

# Repository Information
Here, we briefly outline the structure of the repository. Feel free to get in touch to ask any questions.
- `ConnectivityTests`: Pre-requisites exercises and other miscellenea to ensure connection with Apollo.
  - `Client`: Client-side code to be run on local machines.
  - `Server`: Server-side code to be ran on Apollo.
  - `Utils`: Useful code methods, including getting and setting parameters, and ML code.
  - `.misc`: Preparatory stuff (_has no practical purpose anymore_).

__*Note:*__ _The above cannot be run until I have access to the Apollo machine._

# How to run

### Pre-Requisites
In order to run the code, you will first need to clone the repository, and install the following dependencies from the commandline, via `pip`:
```bash
pip install matplotlib
pip install numpy 
pip install torch
pip install torchvision
pip install flwr
pip install flwr_datasets
```
### Server Information

Execution depends on whether you are on the server or client side. The server side code is to be run on Imperial Apollo, and the client side code is to be run on your local machine.

_**Note crucially that the server side will ONLY be able to run on Imperial's Apollo server. In order to configure this to run on your own machine, change the `server_address` parameter of both the client and the server to your own (this can be found by running `ifconfig` on a Linux machine, and then choosing an available port).**_

### Connectivity Tests

Before any sort of execution can occur, first navigate to the `ConnectivityTests` directory.

On the server-side, simply navigate to the Server directory and run the following command from cmd:
```bash
python .\main.py
```
The server will remain on until federated learning has occured for the number of epochs specified in the `main.py` file. On the client side, navigate to the Client directory and run the following command from cmd, and for the number of clients you wish to run:

If you are running as a client, navigate to the Client directory and run the following command from cmd:
```bash
python .\main.py
```

# Dependencies & Useful Links
### Dependencies
- __Flower__: [Federated Learning Framework.][2]
- __Torch__: [Machine Learning API.][3]
### Links

# Authors
- Rahul George, [rg922@ic.ac.uk][1]

[1]: rg922@ic.ac.uk
[2]: https://flower.ai/docs/framework/index.html
[3]: https://pytorch.org/tutorials/
[4]: https://www.nature.com/articles/s41591-018-0213-5
