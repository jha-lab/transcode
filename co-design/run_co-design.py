# Run co-design of the transformer and the accelerator

import os
import gc
import re
import sys
import json
import yaml
import time
import pickle
import shutil
import hashlib
import argparse
import itertools
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from src.opt import *
from src.utils import *

sys.path.append('../surrogate_modeling/utils')

import embedding_util

sys.path.append('../../txf_design-space/flexibert')

from embeddings.utils import graph_util

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

# Define max measures for co-design
MAX_GLUE = 1
MAX_AREA = 30000 # in mm2
MAX_DYNAMIC_ENERGY = 9e10 # in nJ
MAX_LEAKAGE_ENERGY = 2e10 # in nJ
MAX_CYCLES = 4e9

# Define hyperparameters for co-design
K_GLUE = 0.5
K_AREA = 0.1
K_DYNAMIC_ENERGY = 0.2
K_LEAKAGE_ENERGY = 0.1
K_LATENCY = 0.1

PERFORMANCE_PATIENCE = 10
RANDOM_SAMPLES = 100
K = 4

LOAD_FROM_PRETRAINED = True
MIN_PERFORMANCE = -1
NUM_CORES = mp.cpu_count()


def main(args):

	mode = get_mode(args)

	output_file = f'./{mode}/{args.alg}/results.json'
	os.makedirs(f'./{mode}/{args.alg}/', exist_ok=True)

	results = {}
	if os.path.exists(output_file):
		results = json.load(open(output_file, 'r'))

	if args.alg == 'boshcode':
		results['boshcode'] = boshcode(args)
	else:
		assert mode == 'co_des'
		results[args.alg] = standard_search(args)

	json.dump(results, open(output_file, 'w+'))	
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for conversion',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--design_space_acceltran',
		metavar='',
		type=str,
		default='../surrogate_modeling/design_space/design_space_accelerator.yaml',
		help='path to the accelerator design space file')
	parser.add_argument('--design_space_flexibert',
		metavar='',
		type=str,
		default='../surrogate_modeling/design_space/design_space_transformer.yaml',
		help='path to the transformer design space file')
	parser.add_argument('--flexibert_surrogate',
		metavar='',
		type=str,
		default='../surrogate_modeling/dataset/surrogate_models/glue.pkl',
		help='path to the flexibert surrogate model')
	parser.add_argument('--accelerator_dataset',
		metavar='',
		type=str,
		default='../surrogate_modeling/dataset/dataset_accelerator.json',
		help='path to the accelerator dataset')
	parser.add_argument('--constants_path',
		metavar='',
		type=str,
		default='../acceltran/simulator/constants/constants.yaml',
		help='path to the accelerator simulation constants file')
	parser.add_argument('--acc_embedding',
		metavar='',
		type=str,
		default=None,
		help='accelerator embedding to run HW-NAS')
	parser.add_argument('--txf_embedding',
		metavar='',
		type=str,
		default=None,
		help='transformer embedding to run accelerator search')
	parser.add_argument('--alg',
		metavar='',
		type=str,
		default='boshcode',
		help='algorithm for optimization')
	
	args = parser.parse_args()

	main(args)

