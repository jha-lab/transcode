# Generate a dataset for accelerator performance on diverse transformer models

# Author : Shikhar Tuli

import os
import gc
import re
import sys
import json
import yaml
import time
import pickle
import shutil
import argparse
import itertools
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from matplotlib import pyplot as plt

sys.path.append('./utils/')

import embedding_util
import print_util

sys.path.append('../boshnas/boshnas/')

from boshnas import BOSHNAS
from boshnas_2inp import BOSHNAS as BOSHCODE
from acq import gosh_acq as acq

sys.path.append('../acceltran/simulator/src/')

from pe import *
from ops import *
from buffer import *
from modules import *
from tiled_ops import *
from simulator import *
from accelerator import *
from dict2ops import main as dict2ops
from tqdm.notebook import tqdm


def predict_fn(gbdtr, X):
    """Prepare stochastic surrogate model"""
    mean = float(gbdtr.predict(X))
    dt_preds = []

    for estimator in gbdtr.estimators_:
        pred = estimator[0].predict(X)
        dt_preds.append(pred)
    std = float(np.std(np.array(dt_preds)))

    return float(np.random.normal(mean, std))


def simulate_pair(samples: dict, sample: str, design_space_flexibert: dict, design_space_acceltran: dict, constants: dict, gbdtr):
    """Simulate transformer-accelerator pair"""
    txf_samples = embedding_util.get_samples(design_space_flexibert, 16, 'Random', space='transformer')

    dataset = {}
    
    for txf_sample in txf_samples:
        logs_dir = f'./temp/{sample}_{txf_sample}/'
        os.makedirs(os.path.join(logs_dir, 'metrics'))

        failed_attempt = False

        while True:
            if not failed_attempt:
                model_dict = txf_samples[txf_sample]['model_dict']
            else:
                txf_samples_new = embedding_util.get_samples(design_space_flexibert, 1, 'Random', space='transformer')
                model_dict = txf_samples_new[list(txf_samples_new.keys())[0]]['model_dict']
            try:
                logs = simulate_fast(model_dict,
                                 embedding_util.embedding_to_config(samples[sample], design_space_acceltran),
                                 constants, 
                                 design_space_acceltran,
                                 logs_dir, 
                                 1000, 
                                 debug=False)
            except RuntimeError:
                failed_attempt = True
            else:
                break
        
        glue_score = predict_fn(gbdtr, np.array(txf_samples[txf_sample]['embedding']).reshape(1, -1))
        
        dataset[f'{sample}_{txf_sample}'] = {'model_dict': txf_samples[txf_sample]['model_dict'], 
                                             'txf_embedding': list(map(int, txf_samples[txf_sample]['embedding'])),
                                             'glue_score': glue_score,
                                             'acc_embedding': list(map(int, samples[sample])),
                                             'logs_fast': logs}

    os.makedirs('./dataset/dataset_accelerator/', exist_ok=True)
    json.dump(dataset, open(f'./dataset/dataset_accelerator/{sample}.json', 'w+'))

    return dataset


def main():
    """Generate dataset"""
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--txf_surrogate_file',
        metavar='', 
        type=str, 
        default='./dataset/surrogate_models/glue.pkl',
        help='path to .pkl surrogate model file for the transformer space')
    parser.add_argument('--txf_design_space_file',
        metavar='', 
        type=str, 
        default='./design_space/design_space_transformer.yaml',
        help='path to transformer design space file')
    parser.add_argument('--acc_design_space_file',
        metavar='', 
        type=str, 
        default='./design_space/design_space_accelerator.yaml',
        help='path to transformer design space file')
    parser.add_argument('--constants_file',
        metavar='',
        type=str,
        default='../acceltran/simulator/constants/constants.yaml',
        help='path to the constants.yaml file')
    parser.add_argument('--num_processes',
        metavar='',
        type=int,
        default=16,
        help='number of parallel processes')
    parser.add_argument('--dataset_file',
        metavar='',
        type=str,
        default='./dataset/dataset_accelerator.json',
        help='path to store the dataset')

    args = parser.parse_args()

    if os.path.exists('./temp/') and os.path.isdir('./temp/'):
        shutil.rmtree('./temp/')

    # Load design spaces
    design_space_acceltran = yaml.safe_load(open(args.acc_design_space_file))
    design_space_flexibert = yaml.safe_load(open(args.txf_design_space_file))

    # Load FlexiBERT 2.0 surrogate model
    gbdtr = pickle.load(open(args.txf_surrogate_file, 'rb'))

    constants = yaml.safe_load(open(args.constants_file))

    dataset_list = []
    if os.path.exists('./dataset/dataset_accelerator/') and not os.listdir('./dataset/dataset_accelerator/'):
        for sample_file in os.listdir('./dataset/dataset_accelerator/'):
            dataset_list.append(json.load(open(os.path.join('./dataset/dataset_accelerator/', sample_file))))

    samples = embedding_util.get_samples(design_space_acceltran, 16 - len(dataset_list), 'Random', space='accelerator')

    print(f'Running evaluation on {len(samples)} accelerators')

    # Get performance for transformer-accelerator pairs
    with mp.Pool(processes=args.num_processes) as pool:
        dataset_list.extend(pool.starmap(simulate_pair, [(samples, sample, design_space_flexibert, design_space_acceltran, constants, gbdtr) for sample in samples.keys()]))

    dataset = {}
    for sample_dict in dataset_list:
        dataset.update(sample_dict)

    json.dump(dataset, open(args.dataset_file, 'w+'))

    shutil.rmtree('./temp/')


if __name__ == '__main__':
    main()

