"""
Define the parameters to feed to train_main.py
"""

import argparse

# Define a custom function for boolean variables
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
  
# Arguments
parser = argparse.ArgumentParser(description='Train a MME.')

# Train arguments
parser.add_argument('--data', help='Dataset: one of: "f8k", "f30k", "coco"',
                  default='f8k')
parser.add_argument('--embedding', help='Embedding: one of: "AVGtt_Gfc7","AVGtt_FN_KSBsp0.15n0.25_Gall"',
                  default="AVGtt_FN_KSBsp0.15n0.25_Gall")
parser.add_argument('--loss', help='Loss function to use: one of: "SH", "MH", "OE", "MOE"',
                  default="SH")
parser.add_argument('--margin', type=float, help='Margin for contrastive loss in [0,1]',
                  default=0.2)
parser.add_argument('--dim', type=int, help='Dimensionality of resulting embedding',
                  default=300)
parser.add_argument('--dim_image', type=int, help='Dimensionality of image embedding',
                  default=12416)
parser.add_argument('--dim_word', type=int, help='Dimensionality of word embedding',
                  default=300)
parser.add_argument('--encoder', help='Sentence encoder: "gru" or "bow"',
                  default='gru')
parser.add_argument('--max_epochs', type=int, help='Max number of training epochs',
                  default=30)
parser.add_argument('--dispFreq', type=int, help='Number of samples proccessed before print stats',
                  default=10)
parser.add_argument('--decay_c', type=float, help='',
                  default=0.0)
parser.add_argument('--grad_clip', type=float, help='Maximum module of backpropagation gradients in GRU',
                  default=2.0)
parser.add_argument('--maxlen_w', type=int, help='Maximum number of words in a sentence',
                  default=100)
parser.add_argument('--optimizer', help='Train optimizer: "adam" ',
                  default='adam')
parser.add_argument('--batch_size', type=int, help='Batch size',
                  default=128)
parser.add_argument('--model_name', help='Name for the model saved file ',
                  default='/gpfs/projects/bsc28/MME/order-embedding/train_dir/default')
parser.add_argument('--validFreq', type=int, help='Compute validation every --validFreq batches',
                  default=100)
parser.add_argument('--lrate', type=float, help='Learning rate',
                  default=0.0002)
parser.add_argument('--reload_', type=str2bool, help='Reload existing model for further training',
                  default=False)
parser.add_argument('--data_path', help='Path to data',
                  default='/gpfs/projects/bsc28/MME/data/')
parser.add_argument('--use_glove', type=int, help='If 0, Do not use Glove embedding. One-hot embedding is used instead. If 1, Use a Glove embedding restricting vocabulary to train and validation words. If 2, Use unrestricted Glove embedding. vocabulary_size and vocabulary_min_num_samples are overiden.',
                  default=0)
parser.add_argument('--reduce_lr', type=str2bool, help='Reduce the learning rate after 15 epochs to 1/10 of original',
                  default=False)
parser.add_argument('--abs', type=str2bool, help='Take absolute value of the embeddings. Useful for order embeddings',
                  default=False)
parser.add_argument('--img_norm', type=str2bool, help='Take L2 norm of image embedding. Useful for MH embeddings',
                  default=False)
# Experiment arguments
parser.add_argument('--experiment_name', help='Name to identify the experiment ',
                  default='default')
parser.add_argument('--dataset_name', help='Dataset: one of: "f8k", "f30k", "coco"',
                  default='f8k')

# New, OE arguments
parser.add_argument('--load_from', help='path to the file where model to load is saved ',
                  default=None)
parser.add_argument('--save_dir', help='dir where model trained is saved ',
                  default='default')
parser.add_argument('--method', help='method to use for the loss. Posible choices are: "order", "cosine" ',
                  default='order')
parser.add_argument('--test_subset', help='Which of the two test and val subsets use for coco. Posible choices are: "1k", "5k"',
                  default='1k')
