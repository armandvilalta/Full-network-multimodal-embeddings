"""
Dataset loading
"""
from __future__ import print_function
import numpy
import paths
import os.path as osp

def load_dataset(dataset_name='f8k', 
                 embedding= 'AVGtt_FN_KSBsp0.15n0.25_Gall',
                 path_to_data = 'path_to_data',
                 test_subset='1k',
                 load_train=True,
                 fold=0):
    """
    Load captions and image features
    Parameters:
    -----------
    dataset_name: str
        Name of the dataset.
        One of: 'f8k', 'f30k', 'coco'.
    embedding: str
        Name of the embedding. 
        One of: 'AVGtt_FN_KSBsp0.15n0.25_Gall' for Full Network Embedding, 'AVGtt_Gfc7' for L2-norm FC7 embedding.
    path_to_data: str
        Path to folder that contains one folder for each dataset.
        Dataset folders inside are named 'f8k', 'f30k', 'coco'.
    test_subset: str
        Test subset to be used.
        One of '1k', '5k'.
    load_train: bool
        If true loads train subset. If False only loads validation and test subsets.
    fold: int
        Number of fold the data is divided in for N-fold cross-validation. One of {0, 1, ... , N-1}

    Returns:
    --------
    dataset: dict(dict())
        Dictionary containing the data. Structured as dataset['train', 'val', 'test']['imgs', 'caps']

    """

    if load_train:
        subsets = ['train', 'val', 'test']
    else:
        subsets = ['val', 'test']

    dataset = {}

    for subset in subsets:
        dataset[subset] = {}
        caps_filename = get_caps_filename(path_to_data, dataset_name, subset)
        imgs_filename = get_imgs_filename(path_to_data, dataset_name, embedding, subset)
        caps = []

        # Load caps
        with open(caps_filename, 'rb') as f:
            for line in f:
                caps.append(line.strip())
            dataset[subset]['caps'] = caps

        # Load imgs
        dataset[subset]['imgs'] = numpy.load(imgs_filename)

        # Dtype conversion
        print ('Original dtype subset', subset, '=', dataset[subset]['imgs'].dtype)
        dataset[subset]['imgs'] = dataset[subset]['imgs'].astype(numpy.float32)
        print ('Converted dtype subset', subset, '=', dataset[subset]['imgs'].dtype)

        # handle coco specially by only taking 1k or 5k captions/images
        if dataset_name == 'coco' and subset in ['val', 'test']:
            if test_subset=='1k':
                k=1
            elif test_subset=='5k':
                k=5
            dataset[subset]['imgs'] = dataset[subset]['imgs'][fold*1000*k:(fold+1)*1000*k]
            dataset[subset]['caps'] = dataset[subset]['caps'][fold*5000*k:(fold+1)*5000*k]

    return dataset


def get_caps_filename(path_to_data, dataset_name, subset):
    embedding_file_name = dataset_name + '_' + subset + '_caps.txt'
    caps_filename = osp.join(path_to_data, dataset_name, embedding_file_name)
    return caps_filename

def get_imgs_filename(path_to_data, dataset_name, embedding_name, subset):
    embedding_file_name = 'vgg16_ImageNet_' + dataset_name + '_C1avg_E_' + embedding_name + '_' + subset + '_.npy'
    ims_filename = osp.join(path_to_data, dataset_name, embedding_file_name)
    return ims_filename