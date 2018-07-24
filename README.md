# Full-network-multimodal-embeddings
Code used in the paper: Studying the Impact of the Full-Network Embedding on Multimodal Pipelines (currently under review)

Similar to [visual-semantic-embedding](https://github.com/ryankiros/visual-semantic-embedding) and  [order-embedding](https://github.com/ivendrov/order-embedding) of which this repository is a fork, 
we map images and their captions into a common vector space.

This version adds the max loss option from F. [Faghri, D.J. Fleet, J.R. Kiros and S. Fidler, VSE++: Improving Visual-Semantic Embeddings with Hard Negatives,
arXiv preprint arXiv:1707.05612 (2017).](https://arxiv.org/abs/1707.05612)

The precomputed image embeddings for the Full Network Embedding and the FC7 embedding can be downloaded from [High Performance Artificial Intelligence Group at Barcelona Supercomputing Center](hpai.bsc.es). The trained models for the results in the paper are available at the same web.

## Dependencies

* [Python 2.7](https://www.python.org/downloads/release/python-2713/)
* [Theano 1.0.0](http://deeplearning.net/software/theano/install.html)
* [Numpy 1.15.0](https://pypi.org/project/numpy/)
* [Scipy 1.1.0](https://www.scipy.org/scipylib/download.html)

## Replicating the paper

* Download the precomputed embeddings from [High Performance Artificial Intelligence Group at Barcelona Supercomputing Center](hpai.bsc.es) and place them in ```data``` folder in the repo.
* Download the trained models from [High Performance Artificial Intelligence Group at Barcelona Supercomputing Center](hpai.bsc.es) and place them in ```trained_models_paper``` folder in the repo.
* Run the launcher of the experiment found in the folder ```launchers_eval```
* If you wish to train the model from scratch you can modify the launcher's first line from ``` python eval_main.py \``` to ``` python train_main.py \```. The code will train a new model and evaluate it.

## Modifying the parameters
A detailed description of all the parameters can be found in ```parameters.py```
### Experiment info parameters
* ```--experiment_name``` Name to identify the experiment.
* ```--dataset_name``` Dataset: one of: "f8k", "f30k", "coco".
* ```--model_name``` Name for the model saved file. The experiments in the paper use ```--model_name = --dataset_name + '_' + --experiment_name```.

### Data parameters
* ```--data``` Dataset: one of: "f8k", "f30k", "coco".
* ```--data_path``` Path to data
- ```--embedding``` Embedding: one of: "AVGtt_Gfc7", "AVGtt_FN_KSBsp0.15n0.25_Gall".
- ```--dim_image``` Dimensionality of image embedding.
    - If ```--embedding = 'AVGtt_Gfc7'``` then ```--dim_image = 4096 ```.
    - If ```--embedding = 'AVGtt_FN_KSBsp0.15n0.25_Gall'``` then ``` --dim_image = 12416 ```.
    
### Embedding parameters
* ```--dim``` Dimensionality of resulting multimodal embedding.
* ```--dim_word``` Dimensionality of trainable word embedding.
* ```--loss``` Loss function to use: one of: "SH", "MH", "OE", "MOE".
* ```--abs``` Take absolute value of the embeddings. Useful for order embedding.
* ```--img_norm``` Take L2 norm of image embedding. Useful for MH embeddings.
* ```--method``` Method to use for the loss. Posible choices are: "order", * "cosine".

### Training parameters
* ```--margin``` Margin for contrastive loss in [0,1].
* ```--max_epochs``` Max number of training epochs.
* ```--dispFreq``` Number of samples proccessed before print stats.
* ```--grad_clip``` Maximum module of backpropagation gradients in GRU.
* ```--batch_size``` Batch size.
* ```--validFreq``` Compute validation every --validFreq batches.
* ```--lrate``` Learning rate.

### Saving / loading parameters
* ```--reload_``` Reload existing model for further training.
* ```--load_from``` Path to the file where model to load is saved.
* ```--save_dir``` Folder where model trained is saved.

### Test parameters
* ```--test_subset``` Which of the two test and val subsets use for coco. Posible choices are: "1k", "5k".

## Reference

If you found this code useful, please cite the following paper:
TODO

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
