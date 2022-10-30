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
import argparse
import itertools
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

sys.path.append('../surrogate_modeling/utils')

import embedding_util

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
MAX_DYNAMIC_ENERGY = 8e9 # in nJ
MAX_LEAKAGE_ENERGY = 2e8 # in nJ
MAX_CYCLES = 60e6

# Define hyperparameters for co-design
K_GLUE = 0.5
K_AREA = 0.1
K_DYNAMIC_ENERGY = 0.2
K_LEAKAGE_ENERGY = 0.1
K_LATENCY = 0.1

PERFORMANCE_PATIENCE = 5
RANDOM_SAMPLES = 10 # 100
K = 2 # 8

LOAD_FROM_PRETRAINED = True
MIN_PERFORMANCE = -1
NUM_CORES = mp.cpu_count()


def predict_fn(gbdtr, X):
	# Prepare stochastic surrogate model
	mean = float(gbdtr.predict(X))
	dt_preds = []
	for estimator in gbdtr.estimators_:
		pred = estimator[0].predict(X)
		dt_preds.append(pred)
	std = float(np.std(np.array(dt_preds)))
	return float(np.random.normal(mean, std))


def embedding_to_area(embedding, design_space, constants):
	# Get area from accelerator embedding
	config = embedding_util.embedding_to_config(embedding, design_space)
	accelerator = Accelerator(config, constants)
	return accelerator.area / 1e6


def dict_to_energies(energy_dict):
	# Get dynamic and leakage energies from energy dictionary
	dynamic_energy = energy_dict['buffer'] + energy_dict['main_memory'] + energy_dict['mac_lanes'][0] \
		+ energy_dict['softmax'][0] + energy_dict['layer_norm'][0] + energy_dict['sparsity'][0] \
		+ energy_dict['others'][0]
	leakage_energy = energy_dict['mac_lanes'][1] + energy_dict['softmax'][1] + energy_dict['layer_norm'][1] \
		+ energy_dict['sparsity'][1] + energy_dict['others'][1]
	return dynamic_energy, leakage_energy


def simulate_pair(txf_embedding, acc_embedding, txf_hash, acc_hash, txf_model_dict, acc_config, gbdtr, constants, design_space_acceltran):
	glue_score = predict_fn(gbdtr, txf_embedding.reshape(1, -1))

	results_file_path = f'./logs/{acc_hash}_{txf_hash}/metrics/results.json'
	if os.path.exists(results_file_path):
		return json.load(open(results_file_path))

	os.makedirs(f'./logs/{acc_hash}_{txf_hash}/metrics', exist_ok=True)
	try:
		logs = simulate_fast(model_dict=txf_model_dict, 
							 config=acc_config,
							 constants=constants,
							 design_space=design_space_acceltran,
							 logs_dir=f'./logs/{acc_hash}_{txf_hash}/',
							 plot_steps=10000)

		dataset = {}
		dataset[f'{acc_embedding}_{txf_embedding}'] = {'model_dict': txf_model_dict, 
													   'txf_embedding': list(map(int, txf_embedding)),
													   'glue_score': glue_score,
													   'acc_embedding': list(map(int, acc_embedding)),
													   'logs_fast': logs}

		json.dump(dataset, open(f'./logs/{acc_hash}_{txf_hash}/metrics/results.json', 'w+'))
		return dataset
	except RuntimeError:
		return {}


def get_performances(dataset_list, dataset):
	# Get performances from the given dataset list and update main dataset dictionary
	performances = []

	for sample_dict in dataset_list:
		if sample_dict:
			assert len(sample_dict) == 1
			dataset.update(sample_dict)

			results = list(sample_dict.values())[0]
			y_glue_new = results['glue_score']
			logs = results['logs_fast']


			y_latency_new = logs['cycles']
			y_dyn_energy_new, y_leak_energy_new = dict_to_energies(logs['energy'])
			y_area = logs['area']
			
			performance = K_GLUE * (y_glue_new / MAX_GLUE) + \
						  K_LATENCY * (1 - y_latency_new / MAX_CYCLES) + \
						  K_DYNAMIC_ENERGY * (1 - y_dyn_energy_new / MAX_DYNAMIC_ENERGY) + \
						  K_LEAKAGE_ENERGY * (1 - y_leak_energy_new / MAX_LEAKAGE_ENERGY)
		else:
			performance = MIN_PERFORMANCE

		performances.append(performance)

	return performances, dataset


def main(args):
	# Load accelerator and transformer design spaces
	design_space_acceltran = yaml.safe_load(open(args.design_space_acceltran))
	design_space_flexibert = yaml.safe_load(open(args.design_space_flexibert))

	# Get embedding bounds
	bounds_txf = embedding_util.get_embedding_bounds(design_space_flexibert, 'all', 'transformer')
	bounds_txf = (np.array([bound[0] for bound in bounds_txf]), \
						np.array([bound[1] for bound in bounds_txf]))

	bounds_accel = embedding_util.get_embedding_bounds(design_space_acceltran, 'all', 'accelerator')
	bounds_accel = (np.array([bound[0] for bound in bounds_accel]), \
						np.array([bound[1] for bound in bounds_accel]))

	# Instantiate the BOSHCODE surrogate
	surrogate_model = BOSHCODE(input_dim1=37, # transformer embeddings
							   input_dim2=12, # accelerator embeddings
							   bounds1=bounds_txf,
							   bounds2=bounds_accel,
							   trust_region=False,
							   second_order=True,
							   parallel=True,
							   model_aleatoric=True,
							   save_path='./dataset/surrogate_model/',
							   pretrained=LOAD_FROM_PRETRAINED)

	# Load FlexiBERT 2.0 surrogate model
	gbdtr = pickle.load(open(args.flexibert_surrogate, 'rb'))

	# Load accelerator simulation constants
	constants = yaml.safe_load(open(args.constants_path))

	# Get maximum values from the dataset
	dataset = json.load(open(args.accelerator_dataset))

	max_area, max_dynamic_energy, max_leakage_energy, max_cycles = 0, 0, 0, 0

	for pair_hash in dataset.keys():
		if dataset[pair_hash]['logs_fast']['area'] > max_area: max_area = dataset[pair_hash]['logs_fast']['area']
		if dataset[pair_hash]['logs_fast']['cycles'] > max_cycles: max_cycles = dataset[pair_hash]['logs_fast']['cycles']
			
		energy_dict = dataset[pair_hash]['logs_fast']['energy']
		dynamic_energy, leakage_energy = dict_to_energies(energy_dict)
		
		if dynamic_energy > max_dynamic_energy: max_dynamic_energy = dynamic_energy
		if leakage_energy > max_leakage_energy: max_leakage_energy = leakage_energy

	assert MAX_AREA > max_area
	assert MAX_DYNAMIC_ENERGY > max_dynamic_energy 
	assert MAX_LEAKAGE_ENERGY > max_leakage_energy 
	assert MAX_CYCLES > max_cycles

	# Convert dataset to tabular format
	X_txf, X_acc, y_glue, y_area, y_dyn_energy, y_leak_energy, y_latency = [], [], [], [], [], [], []

	for pair_hash, measures in dataset.items():
		X_txf.append(measures['txf_embedding'])
		X_acc.append(measures['acc_embedding'])
		y_glue.append(measures['glue_score'])
		y_area.append(measures['logs_fast']['area'])
		
		dynamic_energy, leakage_energy = dict_to_energies(measures['logs_fast']['energy'])
		y_dyn_energy.append(dynamic_energy); y_leak_energy.append(leakage_energy)
		
		y_latency.append(measures['logs_fast']['cycles'])
		
	X_txf, X_acc = np.array(X_txf), np.array(X_acc)

	# Get convex combination output
	y = []
	for i in range(len(y_glue)):
		performance = K_GLUE * (y_glue[i] / MAX_GLUE) + \
					  K_LATENCY * (1 - y_latency[i] / MAX_CYCLES) + \
					  K_DYNAMIC_ENERGY * (1 - y_dyn_energy[i] / MAX_DYNAMIC_ENERGY) + \
					  K_LEAKAGE_ENERGY * (1 - y_leak_energy[i] / MAX_LEAKAGE_ENERGY)
		y.append(1 - performance)
		
	y = np.array(y)

	max_loss = np.amax(y)
	y = y / max_loss

	if os.path.exists('./dataset/boshcode.npz'):
		np_load = np.load('./dataset/boshcode.npz')
		X_txf = np_load['X_txf']
		X_acc = np_load['X_acc']
		y = np_load['y']
		max_loss = np_load['max_loss']

	print(f'Current iteration: 0. \tBest performance: {np.amin(y)}')

	# Train surrogate model
	if not LOAD_FROM_PRETRAINED:
		train_error = surrogate_model.train(X_txf, X_acc, y)

	# Run co-design with exploration on random samples
	best_performance, old_best_performance, same_performance, itn = np.amin(y), np.inf, 0, 0
	while same_performance < PERFORMANCE_PATIENCE:
		random_txf_sample_dicts = embedding_util.get_samples(design_space_flexibert, 
												num_samples=RANDOM_SAMPLES, 
												sampling_method='Random', 
												space='transformer',
												debug=False)
		random_txf_samples = [random_txf_sample_dicts[model]['embedding'] for model in random_txf_sample_dicts.keys()]
		
		random_acc_sample_dicts = embedding_util.get_samples(design_space_acceltran, 
												num_samples=RANDOM_SAMPLES, 
												sampling_method='Random', 
												space='accelerator',
												debug=False)
		random_acc_samples = [random_acc_sample_dicts[acc] for acc in random_acc_sample_dicts.keys()]
		
		random_samples = [(np.array(random_txf_samples[i]), 
						   np.array(random_acc_samples[i])) for i in range(RANDOM_SAMPLES)]
		
		# Get queries using GOBI
		query_indices = surrogate_model.get_queries(x=random_samples, k=K, explore_type='ucb', use_al=True)

		# Get performances parallely for all queries
		with mp.Pool(processes=NUM_CORES) as pool:
			dataset_list = pool.starmap(simulate_pair, [(random_samples[i][0],
														   random_samples[i][1],
														   list(random_txf_sample_dicts.keys())[i],
														   list(random_acc_sample_dicts.keys())[i],
														   random_txf_sample_dicts[list(random_txf_sample_dicts.keys())[i]]['model_dict'],
														   embedding_util.embedding_to_config(random_samples[i][1].tolist(), design_space_acceltran),
														   gbdtr,
														   constants,
														   design_space_acceltran) for i in set(query_indices)])

		performances, dataset = get_performances(dataset_list, dataset)

		for i, performance in enumerate(performances):
			y_new = np.array([1 - performance])
			print(f'Performance of queried pair: {y_new / max_loss}')
			
			y = np.concatenate((y, y_new / max_loss))

			X_txf_new, X_acc_new = random_samples[list(set(query_indices))[i]]
			
			X_txf, X_acc = np.concatenate((X_txf, X_txf_new.reshape(1, -1))), \
				np.concatenate((X_acc, X_acc_new.reshape(1, -1)))
		
		best_performance = np.amin(y)
		print(f'Current iteration: {itn + 1}. \tBest performance: {best_performance}')
		itn += 1
		
		# Update same_performance to check convergence
		if best_performance == old_best_performance:
			same_performance += 1

		old_best_performance = best_performance
		
		# Train model on expanded dataset
		train_error = surrogate_model.train(X_txf, X_acc, y)

		# Save expanded dataset
		np.savez('./dataset/boshcode.npz', X_txf=X_txf, X_acc=X_acc, y=y, max_loss=max_loss)
		json.dump(dataset, open('./dataset/dataset_boshcode.json', 'w+'))

	# Get the details on the best pair
	best_txf_embedding = X_txf[np.argmin(y), :]
	best_acc_embedding = X_acc[np.argmin(y), :]

	best_acc_config = embedding_util.embedding_to_config(best_acc_embedding, design_space_acceltran)
	best_txf_model_dict = embedding_util.embedding_to_model_dict(best_txf_embedding, design_space_flexibert)

	print(f'Best accelerator configuration:\n{best_acc_config}')
	print(f'Best transformer model:\n{best_txf_model_dict}')
	

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
	
	args = parser.parse_args()

	main(args)

