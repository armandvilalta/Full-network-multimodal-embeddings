"""
This file includes several functions used to manage experiments.
It includes:
    The functiones needed to write the results of the experiment in a csv file.
    The functions managing the names of the:
        options.pkl file: contains all the parameters of the current model except the weights of the trained model.
        solution.pkl file: constains the information required to resume an interrupted experiment and the results.
        model.npz file: contains the weights of the trained model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
from collections import OrderedDict

try:
    print ('Importing pickle from six')
    from six.moves import cPickle as pkl
except:
    print('Import from six failed')
    try:
        print('Importing from cPickle')
        import cPickle as pkl
    except:
        print ('Import native python pickle')
        import pickle as pkl

import csv
from datetime import datetime, timedelta

def translate_flags(flags_in):
    """
    For compatibility of names with tensorflow code, the names of the parameters are not the same.
    In the autolauncher file the names are translated accordingly
    Here the names are translated back to the original names in the experiment.csv file

    Parameters:
    -----------
    flags_in: OrderedDict
        Dictionary containing all the parameters from the theano code

    Returns:
    --------
    flags: OrderedDict
        Dictionary containing all the parameters according to the experiment.csv file
    """

    flags = OrderedDict()

    # Experiment flags
    flags['experiment_name'] = flags_in['experiment_name']
    flags['dataset_name']    = flags_in['data']

    # Train flags     
    flags['embedding_name']  = flags_in['embedding']  
    flags['loss']            = flags_in['loss']      

    flags['margin']          = flags_in['margin']     
    flags['gru_num_units']   = flags_in['dim']        
    flags['imgs_emb_len']    = flags_in['dim_image']  
    flags['vocabulary_size'] = flags_in['dim_word']   
    flags['max_epochs']      = flags_in['max_epochs'] 
    flags['grad_clip']       = flags_in['grad_clip']  
    flags['maxlen_w']        = flags_in['maxlen_w']   
    flags['batch_size']      = flags_in['batch_size'] 
    flags['n_vervose']       = flags_in['validFreq']  
    flags['learning_rate']   = flags_in['lrate']      
    flags['reload_']         = flags_in['reload_']    
    flags['data_path']       = flags_in['data_path']
    #flags['reduce_lr']       = flags_in['reduce_lr']
    flags['abs_val']         = flags_in['abs']
    flags['load_from']       = flags_in['load_from']         #ok
    flags['save_dir']        = flags_in['save_dir']                #ok
    flags['test_subset']     = flags_in['test_subset'] 
    try:    
        flags['use_glove']  = flags_in['use_glove']
    except:
        flags['use_glove']  = 0

    ## Composed flags
    flags['model_name']      = flags_in['model_name']
    flags['img_norm']        = flags_in['img_norm']          #ok
    #
    ## Extra, default, flags
    #flags['capt_preproces'] = 'theano'
    #flags['img_norm'] = 'theano'
    #flags['use_Glove'] = 'theano'
    #flags['shuffle_buffer'] = 'theano'
    #flags['vocabulary_min_num_samples'] = 'theano'
    #flags['n_caps_per_img'] = 'theano'
    #flags['eval_criteria'] = 'best_avg_recall'
    #flags['vervose'] = 'theano'

    return flags

def write_results_file(filename, flags, params):
    """
    Write all parameters of the experiments and results in a csv file

    Parameters:
    -----------
    filename: str
        Name of the results file
    flags: OrderedDict
        Dictionary containing all the parameters according to the experiment.csv file
    params: OrderedDict
        Dictionary containing all the results obtained as output of the train/evaluate execution
        The keys are: 'best_val_score', 'epoch', 'update', 'samples_seen', 'time_until_results',
                      'time_computing', 'best_test_res', 'best_val_res'

    Returns:
    --------
    None

    """
    # Prepare headers and write them if needed
    headers_exp = ['experiment_name', 'dataset_name']
    headers_job = ['job_id', 'train_time']
    headers_flags = [k for k in flags]
    headers_params = ['best_val_score', 'epoch', 'update', 'samples_seen', 'time_until_results']
    headers_res0 = ['cap_ret r1', 'cap_ret r5', 'cap_ret r10', 'cap_ret medr','img_ret r1','img_ret r5','img_ret r10','img_ret medr']
    headers_res = (['test ' + h for h in headers_res0] + 
                   ['val ' + h for h in headers_res0])
    headers = headers_exp + headers_res + headers_params + headers_job + headers_flags

    if not os.path.isfile(filename):
        with open(filename, 'wb') as cvsfile:
            w = csv.writer(cvsfile)
            print (headers)
            w.writerow(headers)

    # Prepare experiment data and write it down
    try:
        slurm_id = os.environ.get('SLURM_JOB_ID')
        job_id = ('tr_{experiment_name}_' + str(slurm_id)).format(**flags)
    except:
        job_id = None
        pass
    line_exp = [flags[k] for k in headers_exp]
    line_job = [job_id, params['time_computing']]
    line_flags = [v for v in flags.values()]
    line_params = [params[k] if k in params else '-' for k in headers_params ]
    line_res = ([x for v in params['best_test_res'].values() for x in v.values()] +
                [x for v in params['best_val_res'].values() for x in v.values()])
    line = line_exp + line_res + line_params + line_job + line_flags

    with open(filename, 'ab') as cvsfile:
        w = csv.writer(cvsfile)
        print (line)
        w.writerow(line)

    return

def get_opt_filename(model_options, previous=False):
    """
    Gets the name of the options file.
    The file containg all the parameters of the current model except the weights of the trained model.
    Files are saved with consecutive numbering.
    It reads the saved files and propose a name for the new file or,
    if previous=True returns the names of the latest saved file

    Parameters:
    -----------
    model_options: OrderedDict
        Dictionary containing the keys 'save_dir' and 'model_name',
        from which values is defined the name of the file
    previous: bool
        If True: returns the latest saved file
        if False: returns a new, consecutive, file name

    Returns:
    --------
    f_prev or f_curr: str
        filename for options.pkl file
    """

    # Find the last file saved
    idx = 0
    while True:
        f = os.path.join( model_options['save_dir'], model_options['model_name'] + '_{0:02.0f}_opt.pkl'.format(idx))
        if os.path.exists(f):
            f_prev = f
            idx +=1
        else:
            f_curr = f
            break
    # Return the name of the last or new file
    if previous:
        try:
            return f_prev
        except:
            print ("Doesn't exist file", f)
            sys.stdout.flush()
    else:
        return f_curr

def get_sol_filename(model_options, best=False, previous=False):
    """
    Gets the name of the solution file.
    This file contains the information needed to resume an interrupted experiment and the results.
    Files are saved with consecutive numbering.
    It reads the saved files and propose a name for the new file or,
    if previous=True returns the names of the latest saved file

    Parameters:
    -----------
    model_options: OrderedDict
        Dictionary containing the keys 'save_dir' and 'model_name',
        from which values is defined the name of the file
    best: bool
        If True: returns the filename of the model that obtained the best results on validation test 
        if False: returns the filename of the model at the end of training.
    previous: bool
        If True: returns the latest saved file
        if False: returns a new, consecutive, file name

    Returns:
    --------
    f_prev or f_curr: str
        filename for solution.pkl file
    """

    # Find the last file saved
    idx = 0
    while True:
        if best:
            f = os.path.join( model_options['save_dir'], model_options['model_name'] + '_BEST_{0:02.0f}_sol.pkl'.format(idx))
        else:
            f = os.path.join( model_options['save_dir'], model_options['model_name'] + '_LAST_{0:02.0f}_sol.pkl'.format(idx))
        if os.path.exists(f):
            f_prev = f
            idx +=1
        else:
            f_curr = f
            break
    # Return the name of the last or new file
    if previous:
        try:
            return f_prev
        except:
            print ("Doesn't exist file", f)
            sys.stdout.flush()
    else:
        return f_curr

    
def get_npz_filename(model_options, best=False, previous=False):
    """
    Gets the name of the saved model file.
    This file contains the weights that define the model for the neural network.
    Files are saved with consecutive numbering.
    It reads the saved files and propose a name for the new file or,
    if previous=True returns the names of the latest saved file

    Parameters:
    -----------
    model_options: OrderedDict
        Dictionary containing the keys 'save_dir' and 'model_name',
        from which values is defined the name of the file
    best: bool
        If True: returns the filename of the model that obtained the best results on validation test 
        if False: returns the filename of the model at the end of training.
    previous: bool
        If True: returns the latest saved file
        if False: returns a new, consecutive, file name

    Returns:
    --------
    f_prev or f_curr: str
        filename for model.npz file
    """

    # Find the last file saved
    idx = 0
    while True:
        if best:
            f = os.path.join( model_options['save_dir'], model_options['model_name'] + '_BEST_{0:02.0f}.npz'.format(idx))
        else:
            f = os.path.join( model_options['save_dir'], model_options['model_name'] + '_LAST_{0:02.0f}.npz'.format(idx))
        if os.path.exists(f):
            f_prev = f
            idx +=1
        else:
            f_curr = f
            break
    # Return the name of the last or new file
    if previous:
        try:
            return f_prev
        except:
            print ("Doesn't exist file", f)
            sys.stdout.flush()
    else:
        return f_curr
