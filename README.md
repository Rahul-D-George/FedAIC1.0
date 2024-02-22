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

## Pre-Requisites

### Server Information

Execution depends on whether you are on the server or client side. The server side code is to be run on Imperial Apollo, and the client side code is to be run on your local machine.

_**Note crucially that the server side will ONLY be able to run on Imperial's Apollo server. In order to configure this to run on your own machine, change the `server_address` parameter of both the client and the server to your own (this can be found by running `ifconfig` on a Linux machine, and then choosing an available port).**_

### Installation Guide
In order to run the code, you will first need to clone the repository, and following that, ensure that you have all the correct dependencies. A small description is provided on how to ensure this.

1. Install the latest Python version from [here][5]. **Ensure to add Python to your PATH and ensure that pip is correctly installed as well**.
2. Once you have Python installed, you can install the required dependencies by running the following command from the root directory of the repository:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all the required dependencies for the project.
3. Once you have installed the dependencies, you can then run the code as described below.

## Connectivity Tests

The connectivity testing folder uses a simple classification task to ensure that the server and client are able to communicate with each other (and more importantly, exchange parameters meaningfully). In particular, the CIFAR-10 dataset is used for this purpose, along with a CNN model. The details of these can be found in the `Utils` subdirectory. The classification task is run on the server, after which the initial model parameters are sent to the client. The client then runs the model on its own data and sends the results back to the server. The server then aggregates the results and confirms the new accuracy of the model.

### Running the Server
Before any sort of execution can occur, first navigate to the `ConnectivityTests` directory.

On the server-side, simply navigate to the `Server` directory and run the following command from cmd:
```bash
python .\main.py
```
<del>The server will remain on until federated learning has occured for the number of epochs specified in the `main.py` file</del>. As of the latest commit, the server will remain on **permanently**, unless some error occurs, or until manually stopped.

### Running the Client

If you are running as a client, navigate to the `Client` directory and run the following command from cmd:
```bash
python .\main.py
```

_Note: If either of these do not work with the `python` command, instead try using `python3` or `py`, as it can vary depending on your OS._

# Dependencies & Useful Links
### Dependencies
- __Flower__: [Federated Learning Framework.][2]
- __Torch__: [Machine Learning API.][3]
### Links
- _Stub_
# Authors
- Rahul George, [rg922@ic.ac.uk][1]

[1]: rg922@ic.ac.uk
[2]: https://flower.ai/docs/framework/index.html
[3]: https://pytorch.org/tutorials/
[4]: https://www.nature.com/articles/s41591-018-0213-5
[5]: https://www.python.org/downloads/