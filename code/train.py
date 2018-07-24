"""
Main trainer function
"""
from __future__ import print_function
import theano

import cPickle as pkl
from collections import OrderedDict

import os
import sys
import time
from datetime import datetime, timedelta

# Local dependencies
import datasource

from utils import *
from optim import adam
from model import  (init_params,
                    build_model,
                    build_sentence_encoder,
                    build_image_encoder,
                    build_errors)
from vocab import build_dictionary
from evaluation import t2i, i2t, build_results_dict
from tools import encode_sentences, encode_images, compute_errors
from datasets import load_dataset
from exp_management import get_opt_filename, get_sol_filename, get_npz_filename

# main trainer
def trainer(**kwargs):
    """
    Train the model according to input params
    Info about input params is available in parameters.py
    """
    # Timing
    print('Starting time:', datetime.now())
    sys.stdout.flush()
    t_start_train = time.time()

    # Model options
    # load old model, including parameters, but overwrite with new options

    # Extract model options from arguments
    model_options = {}
    for k, v in kwargs.iteritems():
        model_options[k] = v

    # Print input options
    print ('PARAMETERS BEFORE LOADING:')
    for k,v in model_options.items():
        print('{:>26}: {}'.format(k,v))
    sys.stdout.flush()

    # Reload options if required
    curr_model = dict()
    if model_options['reload_']:
        # Reload model parameters
        opt_filename_reload = get_opt_filename(model_options, previous=True)
        print ( 'reloading...', opt_filename_reload)
        sys.stdout.flush()
        try:
            with open(opt_filename_reload, 'rb') as f:
                curr_model = pkl.load(f)
        except:
            print ('Failed to reload parameters, try to use only feeded parameters')
            curr_model['options'] = {}
 
        # Check if we reload from best model or last model
        if model_options['load_from'] in ['Best','best','B','b']:
            load_from_best = True
            print ('Loading from Best saved model in validation results')
        elif model_options['load_from'] in ['Last','last','L','l']:
            load_from_best = False
            print ('Loading from Last saved model')
        else:
            print ('Unkown choice for "load_from" parameter', model_options['load_from'])
            print ('Please choose one of:', ['Best','best','B','b'],['Last','last','L','l'])
            print ('Using Last as default')
            load_from_best = False

        # Reload end-point parameters
        state_filename = get_sol_filename(model_options, best=load_from_best, previous=True)
        print ( 'reloading...', state_filename)
        sys.stdout.flush()
        try:
            with open(state_filename, 'rb') as f:
                state_params = pkl.load(f)
            if load_from_best:
                init_epoch = state_params['epoch']
                solution = state_params
            else:
                init_epoch = state_params['epoch_done'] +1
                solution = state_params['solution']
            best_val_score = solution['best_val_score']
            n_samples = solution['samples_seen']
        except:
            print ('Failed to reload state parameters, starting from 0')
            init_epoch = 0
            best_val_score = 0
            n_samples = 0

    else:
        curr_model['options'] = {}
        init_epoch = 0
        best_val_score = 0
        n_samples = 0

    # Overwrite loaded options with input options
    for k, v in kwargs.iteritems():
        curr_model['options'][k] = v
    model_options = curr_model['options']

    # Print final options loaded
    if model_options['reload_']:
        print ('PARAMETERS AFTER LOADING:')
        for k,v in model_options.items():
            print('{:>26}: {}'.format(k,v))
        sys.stdout.flush()

    # Load training and development sets
    print ( 'Loading dataset')
    sys.stdout.flush()

    dataset = load_dataset(dataset_name=model_options['data'],
                           embedding= model_options['embedding'],
                           path_to_data = model_options['data_path'],
                           test_subset=model_options['test_subset'],
                           load_train=True,
                           fold=0)
    train = dataset['train']
    dev = dataset['val']

    # Create word dictionary
    print ( 'Creating dictionary')
    sys.stdout.flush()
    worddict = build_dictionary(train['caps']+dev['caps'])
    print ( 'Dictionary size: ' + str(len(worddict)))
    sys.stdout.flush()
    curr_model['worddict'] = worddict
    curr_model['options']['n_words'] = len(worddict) + 2

    # save model
    opt_filename_save = get_opt_filename(model_options, previous=False)
    print ('Saving model parameters in', opt_filename_save)
    sys.stdout.flush()
    try:
        os.makedirs(os.path.dirname(opt_filename_save))
    except:
        pass
    pkl.dump(curr_model, open(opt_filename_save, 'wb'))

    # Load data from dataset
    print ( 'Loading data')
    sys.stdout.flush()
    train_iter = datasource.Datasource(train, batch_size=model_options['batch_size'], worddict=worddict)
    dev = datasource.Datasource(dev, worddict=worddict)
    dev_caps, dev_ims = dev.all()

    print ('Building model')
    sys.stdout.flush()
    params = init_params(model_options)

    # reload network parameters, ie. weights
    if model_options['reload_']:
        params_filename = get_npz_filename(model_options, best=load_from_best, previous=True)
        params = load_params(params_filename, params)

    tparams = init_tparams(params)
    inps, cost = build_model(tparams, model_options)

    print ( 'Building sentence encoder')
    sys.stdout.flush()
    inps_se, sentences = build_sentence_encoder(tparams, model_options)
    f_senc = theano.function(inps_se, sentences, profile=False)

    print ( 'Building image encoder')
    sys.stdout.flush()
    inps_ie, images = build_image_encoder(tparams, model_options)
    f_ienc = theano.function(inps_ie, images, profile=False)

    print ( 'Building f_grad...')
    sys.stdout.flush()
    grads = tensor.grad(cost, wrt=itemlist(tparams))

    print ( 'Building errors...')
    sys.stdout.flush()
    inps_err, errs = build_errors(model_options)
    f_err = theano.function(inps_err, errs, profile=False)

    curr_model['f_senc'] = f_senc
    curr_model['f_ienc'] = f_ienc
    curr_model['f_err'] = f_err

    if model_options['grad_clip'] > 0.:
        grads = [maxnorm(g, model_options['grad_clip']) for g in grads]

    lr = tensor.scalar(name='lr')
    print ('Building optimizers...')
    sys.stdout.flush()
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(model_options['optimizer'])(lr, tparams, grads, inps, cost)

    # Get names for the files to save model and solution
    sol_filename_best = get_sol_filename(model_options, best=True, previous=False)
    sol_filename_last = get_sol_filename(model_options, best=False, previous=False)
    params_filename_best = get_npz_filename(model_options, best=True, previous=False)
    params_filename_last = get_npz_filename(model_options, best=False, previous=False)

    print ('PATHS TO MODELS:')
    for filename in [sol_filename_best, sol_filename_last, params_filename_best, params_filename_last]:
        print (filename)
        sys.stdout.flush()
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            pass

    # Start optimization
    print ( 'Optimization')
    sys.stdout.flush()

    uidx = 0
    
    # Timing
    t_start = time.time()
    print ('Starting time:', datetime.now())

    for eidx in range(init_epoch, model_options['max_epochs']):
        t_start_epoch = time.time()
        print ( 'Epoch ', eidx)
        sys.stdout.flush()

        for x, mask, im in train_iter:
            n_samples += x.shape[1]
            uidx += 1

            # Update
            ud_start = time.time()
            cost = f_grad_shared(x, mask, im)
            f_update(model_options['lrate'])
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print ( 'NaN detected')
                sys.stdout.flush()
                return 1., 1., 1.

            if numpy.mod(uidx, model_options['dispFreq']) == 0:
                print ( 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud)
                sys.stdout.flush()

            if numpy.mod(uidx, model_options['validFreq']) == 0:
                print ( 'Computing results...')
                sys.stdout.flush()

                # encode sentences efficiently
                dev_s = encode_sentences(curr_model, dev_caps, batch_size=model_options['batch_size'])
                dev_i = encode_images(curr_model, dev_ims)

                # compute errors
                dev_errs = compute_errors(curr_model, dev_s, dev_i)

                # compute ranking error
                (r1, r5, r10, medr, meanr) = i2t(dev_errs)
                (r1i, r5i, r10i, medri, meanri)= t2i(dev_errs)
                print ( "Text to image (dev set): %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri))
                sys.stdout.flush()
                print ( "Image to text (dev set): %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))
                sys.stdout.flush()
                
                # Score
                val_score = r1 + r5 + r10 + r1i + r5i + r10i
                if val_score > best_val_score:
                    
                    print ('BEST MODEL FOUND')
                    print ('Score:',val_score)
                    print ('Previous best score:',best_val_score)
                    best_val_score = val_score
                    # Join in a results dict
                    results_dict = build_results_dict(r1, r5, r10, medr, r1i, r5i, r10i, medri)
                    
                    # Save parameters
                    print ( 'Saving...', end=' ')
                    sys.stdout.flush()
                    numpy.savez(params_filename_best, **unzip(tparams))
                    print ( 'Done')
                    sys.stdout.flush()
                    
                    # Update solution
                    solution = OrderedDict([
                        ('epoch',eidx),
                        ('update',uidx),
                        ('samples_seen',n_samples),
                        ('best_val_score',best_val_score),
                        ('best_val_res',results_dict),
                        ('time_until_results',str(timedelta(seconds=(time.time() - t_start_train))))
                    ])
                    pkl.dump(solution, open(sol_filename_best, 'wb'))


        print ( 'Seen %d samples'%n_samples)
        sys.stdout.flush()

        # Timing
        t_epoch = time.time() - t_start_epoch
        t_epoch_avg = (time.time() -  t_start) / (eidx+1 - (init_epoch))
        print ('Time for this epoch:', 
            str(timedelta(seconds=t_epoch)), 
            'Average:', 
            str(timedelta(seconds=t_epoch_avg)))
        t_2_complete = t_epoch_avg * (model_options['max_epochs'] - (eidx+1))
        print ('Time since start session:', 
            str(timedelta(seconds=time.time()-t_start)), 
            'Estimated time to complete training:', 
            str(timedelta(seconds=t_2_complete)))
        print ('Current time:', datetime.now())
        sys.stdout.flush()

        # Save current model
        try:
            state_params = OrderedDict([
                ('epoch_done',eidx),
                ('solution',solution)
                ])
        except:
            solution = OrderedDict([
                ('epoch',eidx),
                ('update',uidx),
                ('samples_seen',n_samples),
                ('best_val_score',best_val_score),
                ('time_until_results',str(timedelta(seconds=(time.time() - t_start_train))))
            ])
            state_params = OrderedDict([
                ('epoch_done',eidx),
                ('solution',solution)
                ])
        pkl.dump(state_params, open(sol_filename_last, 'wb'))

        # Save parameters
        print ( 'Saving LAST npz...', end=' ')
        sys.stdout.flush()
        numpy.savez(params_filename_last, **unzip(tparams))
        print ( 'Done')
        sys.stdout.flush()

    return solution

if __name__ == '__main__':
    pass

