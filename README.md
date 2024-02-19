# Fed-AIC1.0
This is a newer version of the AI-Clinician<sup>[\[1\]][4]</sup> from the referenced paper. The old version implements reinforcement learning through a tabular method of maintaining Q-values; we instead use a DQN here.
Further, we make use of federated learning in order to make the process of training more efficient and private.

# Repository Information
Here, we briefly outline the structure of the repository. Feel free to get in touch to ask any questions.
- `ConnectivityTests`: Pre-requisites exercises and other miscellenea to ensure connection with Apollo.
  - `Client`: Client-side code to be run on local machines.
  - `Server`: Server-side code to be ran on Apollo.
  - `Utils`: Useful code methods.
  - `.misc`: Preparatory stuff (_has no practical purpose anymore_).

__*Note:*__ _The above cannot be run until I have access to the Apollo machine._

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