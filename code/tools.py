"""
A selection of functions for encoding images and sentences
"""
import theano

import cPickle as pkl
import numpy

from collections import OrderedDict, defaultdict
from scipy.linalg import norm

from utils import load_params, init_tparams
from model import init_params, build_sentence_encoder, build_image_encoder, build_errors
from exp_management import get_opt_filename, get_npz_filename

def load_model(save_dir, model_name, best=True):
    """
    Load all model components
    Input are only save_dir and model_name since it is assumed that filenames keep convention
    """

    model_options = {}
    model_options['save_dir'] = save_dir
    model_options['model_name'] = model_name

    # Load model
    print 'Loading model'
    opt_filename_reload = get_opt_filename(model_options, previous=True)
    with open(opt_filename_reload, 'rb') as f:
        model = pkl.load(f)

    options = model['options']

    # Load parameters
    print 'Loading model parameters...'
    params = init_params(options)
    params_filename = get_npz_filename(model_options, best=best, previous=True)
    params = load_params(params_filename, params)
    tparams = init_tparams(params)

    # Extractor functions
    print 'Compiling sentence encoder...'
    [x, x_mask], sentences = build_sentence_encoder(tparams, options)
    f_senc = theano.function([x, x_mask], sentences, name='f_senc')

    print 'Compiling image encoder...'
    [im], images = build_image_encoder(tparams, options)
    f_ienc = theano.function([im], images, name='f_ienc')

    print 'Compiling error computation...'
    [s, im], errs = build_errors(options)
    f_err = theano.function([s,im], errs, name='f_err')

    # Store everything we need in a dictionary
    print 'Packing up...'
    model['f_senc'] = f_senc
    model['f_ienc'] = f_ienc
    model['f_err'] = f_err
    return model


def load_model_path(path_to_model):
    """
    Load all model components
    """
    print path_to_model

    # Load model
    print 'Loading model'
    with open(path_to_model + '.pkl', 'rb') as f:
        model = pkl.load(f)

    options = model['options']

    # Load parameters
    print 'Loading model parameters...'
    params = init_params(options)

    params = load_params(path_to_model + '.npz', params)
    tparams = init_tparams(params)

    # Extractor functions
    print 'Compiling sentence encoder...'
    [x, x_mask], sentences = build_sentence_encoder(tparams, options)
    f_senc = theano.function([x, x_mask], sentences, name='f_senc')

    print 'Compiling image encoder...'
    [im], images = build_image_encoder(tparams, options)
    f_ienc = theano.function([im], images, name='f_ienc')

    print 'Compiling error computation...'
    [s, im], errs = build_errors(options)
    f_err = theano.function([s,im], errs, name='f_err')

    # Store everything we need in a dictionary
    print 'Packing up...'
    model['f_senc'] = f_senc
    model['f_ienc'] = f_ienc
    model['f_err'] = f_err
    return model

def encode_sentences(model, X, verbose=False, batch_size=128):
    """
    Encode sentences into the joint embedding space
    """
    features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i, s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model['worddict'][w] if w in model['worddict'] and model['worddict'][w] < model['options']['n_words'] else 1 for w in cc])
            x = numpy.zeros((k+1, len(caption))).astype('int64')
            x_mask = numpy.zeros((k+1, len(caption))).astype('float32')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s
                x_mask[:k+1,idx] = 1.
            
            ff = model['f_senc'](x, x_mask)
            for ind, c in enumerate(caps):
                features[c] = ff[ind]

    return features

def encode_images(model, IM):
    """
    Encode images into the joint embedding space
    """
    return model['f_ienc'](IM)


def compute_errors(model, s, im):
    """
    Computes errors between each sentence and caption
    """
    return model['f_err'](s, im)






