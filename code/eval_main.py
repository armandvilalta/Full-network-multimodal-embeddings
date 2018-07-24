"""
Only evaluate an experiment.
Write results in a file named in line 62, res_filename
"""

#import train, tools, evaluation
from __future__ import print_function
from exp_management import  write_results_file, translate_flags
from train import trainer
from tools import load_model
from evaluation import ranking_eval_Nfold

import argparse
import time
from datetime import datetime, timedelta
import os.path
import sys

# Load parameters
cwd = os.getcwd()
print ('Working in', cwd)
sys.stdout.flush()
if '/home/armand/MME/' in cwd:
    from parameters_LOCAL import parser
    print ("Working in Armand's Laptop")
elif '/gpfs/projects/' in cwd:
    from parameters import parser
    print ("Working in GPFS")
sys.stdout.flush()

args, not_args = parser.parse_known_args()
print ('Arguments not accepted:')
print(not_args)
arg_dict = vars(args).copy()

# Keep only the arguments for train function
train_arg = arg_dict.copy()
del train_arg['dataset_name']

print('Train arguments collected from launcher:')
for k,v in train_arg.items():
    print('{:>26}: {}'.format(k,v))
    sys.stdout.flush()

# Parameters
params = {}
params['best_epoch'] = 0

# TRAIN
t_start = time.time()
params['time_computing'] = time.time() - t_start
# TEST
n_fold = 1
model = load_model(args.save_dir, args.model_name, best=True)
print ('VALIDATION:')
params['best_val_res'], params['best_val_score'] = ranking_eval_Nfold(model, n_fold, subset='val') 
print ('TEST:')
params['best_test_res'], _ = ranking_eval_Nfold(model, n_fold, subset='test')

# WRITE RESULTS
flags = translate_flags(arg_dict)
res_filename = os.path.join('/gpfs/projects/bsc28/MME/order-embedding/train_dir' ,'results_OE_OK.csv')
#res_filename = os.path.join('/home/armand/MME/train/theano' ,'results.csv')

write_results_file(filename=res_filename, flags=flags, params=params)
print ('Results written in',res_filename)
sys.stdout.flush()