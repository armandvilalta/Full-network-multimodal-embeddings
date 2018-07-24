# Full-network-multimodal-embeddings
Code used in the paper: Studying the Impact of the Full-Network Embedding on Multimodal Pipelines (currently under review)

Similar to [visual-semantic-embedding](https://github.com/ryankiros/visual-semantic-embedding) and  [order-embedding](https://github.com/ivendrov/order-embedding) of which this repository is a fork, 
we map images and their captions into a common vector space.

This version adds the max loss option from F. [Faghri, D.J. Fleet, J.R. Kiros and S. Fidler, VSE++: Improving Visual-Semantic Embeddings with Hard Negatives,
arXiv preprint arXiv:1707.05612 (2017).](https://arxiv.org/abs/1707.05612)

The precomputed image embeddings for the Full Network Embedding and the FC7 embedding can be downloaded from [High Performance Artificial Intelligence](hpai.bsc.es). The trained models for the results in the paper are available at the same web.

###Dependencies
*[Python 2.7](https://www.python.org/downloads/release/python-2713/)
*[Theano 1.0.0](http://deeplearning.net/software/theano/install.html)
*[Numpy 1.15.0](https://pypi.org/project/numpy/)
*[Scipy 1.1.0](https://www.scipy.org/scipylib/download.html)
