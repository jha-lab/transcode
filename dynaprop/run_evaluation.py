# Run evaluation on the GLUE tasks with DynaProp training

# Author : Shikhar Tuli

import os
import sys

sys.path.append('../txf_design-space/transformers/src/')
sys.path.append('../txf_design-space/embeddings/')
sys.path.append('../txf_design-space/flexibert/')

import torch
import shlex
import argparse
import re
import json
import math
import numpy as np
from utils.run_glue import main as run_glue
from utils.run_squad import main as run_squad
from utils import run_squad_legacy
import time
import platform

from load_all_glue_datasets import main as load_all_glue_datasets
from datasets import load_dataset, load_metric
from tokenize_glue_datasets import save_dataset
from matplotlib import pyplot as plt

sys.path.append('../txf_design-space/transformers/src/transformers')
from transformers import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular, BertForSequenceClassificationModular
from transformers.models.bert.modeling_dtbert import DTBertModel, DTBertForMaskedLM, DTBertForSequenceClassification, DTBertForQuestionAnswering

import logging
#logging.disable(logging.INFO)
#logging.disable(logging.WARNING)


USE_NON_PRUNED = True
PREFIX_CHECKPOINT_DIR = "checkpoint"
GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
MAX_K = {'sst2': 512, 'squad_v2': 384}
NUM_EPOCHS = 3


def get_training_args(output_dir, task, do_train, train_threshold, sparsity_file):
	if task == 'sst2':
		training_args = f'--model_name_or_path {output_dir} \
			--task_name sst2 \
			--do_eval \
			{"--do_train" if do_train else ""} \
			--num_train_epochs {NUM_EPOCHS} \
			--logging_steps 50 \
			--max_seq_length 512 \
			--overwrite_output_dir \
			--per_device_train_batch_size 64 \
			--per_device_eval_batch_size 64 \
			--dynaprop_min_norm {train_threshold} \
			--dynaprop_json_file {sparsity_file} \
			--output_dir {output_dir}'
	elif task == 'squad_v2':
		training_args = f'--model_name_or_path {output_dir} \
			--dataset_name squad_v2 \
			--version_2_with_negative \
			--do_eval \
			{"--do_train" if do_train else ""} \
			--num_train_epochs {NUM_EPOCHS} \
			--logging_steps 50 \
			--max_seq_length 384 \
			--overwrite_output_dir \
			--per_device_train_batch_size 512 \
			--per_device_eval_batch_size 64 \
			--dynaprop_min_norm {train_threshold} \
			--dynaprop_json_file {sparsity_file} \
			--output_dir {output_dir}'

	training_args = shlex.split(training_args)

	return training_args


def get_tokenizer_args(output_dir, task):

	training_args = f'--task_name {task} \
		--do_eval \
		--max_seq_length 512 \
		--output_dir {output_dir}\
		--overwrite_output_dir'

	training_args = shlex.split(training_args)

	return training_args


def main(args):
	"""Dynamic pruning front-end function"""
	assert args.task in ['sst2', 'squad_v2'], 'Only the SST2 and SQuAD v2 tasks are supported right now'
	assert args.model_name in ['bert-tiny', 'bert-base'], 'Only BERT-Tiny and BERT-Base are supported right now'

	if args.do_train and args.task == 'squad_v2':
		raise ValueError('Training not supported with SQuAD-v2 dataset')

	# Load all GLUE datasets
	load_all_glue_datasets()

	if args.task == 'sst2':
		if args.model_name == 'bert-base':
			if not os.path.exists('./models/bert-base-sst2/pytorch_model.bin'):
				# Load tokenizer and model
				if args.do_train:
					tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
				else:
					tokenizer = BertTokenizer.from_pretrained('howey/bert-base-uncased-sst2')
				tokenizer.save_pretrained('./models/bert-base-sst2/')
				
				# Initialize and save given model
				if args.do_train:
					model = DTBertForSequenceClassification.from_pretrained('bert-base-uncased')
				else:
					model = DTBertForSequenceClassification.from_pretrained('howey/bert-base-uncased-sst2')
				model.save_pretrained('./models/bert-base-sst2/')
			tokenizer = BertTokenizer.from_pretrained('./models/bert-base-sst2/')
			model = DTBertForSequenceClassification.from_pretrained('./models/bert-base-sst2/')
			if not os.path.exists('./models/bert-base-sst2-weight_pruned/pytorch_model.bin'):
				# Load tokenizer and model
				tokenizer = BertTokenizer.from_pretrained('echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid')
				tokenizer.save_pretrained('./models/bert-base-sst2-weight_pruned/')
				
				# Initialize and save given model
				model = DTBertForSequenceClassification.from_pretrained('echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid')
				model.save_pretrained('./models/bert-base-sst2-weight_pruned/')
			tokenizer_wp = BertTokenizer.from_pretrained('./models/bert-base-sst2-weight_pruned/')
			model_wp = DTBertForSequenceClassification.from_pretrained('./models/bert-base-sst2-weight_pruned/')
		elif args.model_name == 'bert-tiny':
			if not os.path.exists('./models/bert-tiny-sst2/pytorch_model.bin'):
				# Load tokenizer and model
				if args.do_train:
					tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
				else:
					tokenizer = BertTokenizer.from_pretrained('philschmid/tiny-bert-sst2-distilled')
				tokenizer.save_pretrained('./models/bert-tiny-sst2/')
				
				# Initialize and save given model
				if args.do_train:
					model = DTBertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny')
				else:
					model = DTBertForSequenceClassification.from_pretrained('philschmid/tiny-bert-sst2-distilled')
				model.save_pretrained('./models/bert-tiny-sst2/')
			tokenizer = BertTokenizer.from_pretrained('./models/bert-tiny-sst2/')
			model = DTBertForSequenceClassification.from_pretrained('./models/bert-tiny-sst2/')
	elif args.task == 'squad_v2':
		if args.model_name == 'bert-base':
			if not os.path.exists('./models/bert-base-squad_v2/pytorch_model.bin'):
				# Load tokenizer and model
				if args.do_train:
					tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
				else:
					tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-uncased-squad2')
				tokenizer.save_pretrained('./models/bert-base-squad_v2/')
				
				# Initialize and save given model
				if args.do_train:
					model = DTBertForSequenceClassification.from_pretrained('bert-base-uncased')
				else:
					model = DTBertForQuestionAnswering.from_pretrained('deepset/bert-base-uncased-squad2')
				model.save_pretrained('./models/bert-base-squad_v2/')
			tokenizer = BertTokenizer.from_pretrained('./models/bert-base-squad_v2/')
			model = DTBertForQuestionAnswering.from_pretrained('./models/bert-base-squad_v2/')
		elif args.model_name == 'bert-tiny':
			if not os.path.exists('./models/bert-tiny-squad_v2/pytorch_model.bin'):
				# Load tokenizer and model
				if args.do_train:
					tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
				else:
					tokenizer = BertTokenizer.from_pretrained('mrm8488/bert-tiny-finetuned-squadv2')
				tokenizer.save_pretrained('./models/bert-tiny-squad_v2/')
				
				# Initialize and save given model
				if args.do_train:
					model = DTBertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny')
				else:
					model = DTBertForQuestionAnswering.from_pretrained('mrm8488/bert-tiny-finetuned-squadv2')
				model.save_pretrained('./models/bert-tiny-squad_v2/')
			tokenizer = BertTokenizer.from_pretrained('./models/bert-tiny-squad_v2/')
			model = DTBertForQuestionAnswering.from_pretrained('./models/bert-tiny-squad_v2/')

	output_dir = os.path.join('./results/' if USE_NON_PRUNED else './results/nn_pruning/', f'{args.model_name}_{args.task}{"_di" if args.max_eval_threshold is not None else ""}{"_dt" if args.max_train_threshold > 0 else ""}{"_wp" if args.prune_weights else ""}')

	print(f'Output directory: {output_dir}')

	os.makedirs(output_dir, exist_ok=True)
	results = []
	if os.path.exists(os.path.join(output_dir, 'results.json')):
		results = json.load(open(os.path.join(output_dir, 'results.json')))

	if args.do_train is True:
		assert args.max_train_threshold is not None

	if args.max_eval_threshold is not None and args.max_train_threshold is None:
		eval_thresholds = list(np.arange(0, args.max_eval_threshold, 0.01))
		train_thresholds = [None for _ in range(len(eval_thresholds))]
	elif args.max_train_threshold is not None and args.max_eval_threshold is None:
		train_thresholds = list(np.arange(0, args.max_train_threshold, 5e-5))
		eval_thresholds = [0 for _ in range(len(train_thresholds))]
	elif args.max_train_threshold is not None and args.max_eval_threshold is not None:
		eval_thresholds, train_thresholds = np.meshgrid(np.arange(0, args.max_eval_threshold, 0.01), np.arange(0, args.max_train_threshold, 5e-5))
		eval_thresholds, train_thresholds = eval_thresholds.reshape(-1).tolist(), train_thresholds.reshape(-1).tolist()
	else:
		raise ValueError(f'Either max_pruning_threshold or min_grad_threshold has to be given')

	k = MAX_K[args.task]
	for i, t in zip(eval_thresholds, train_thresholds):
		print(f'Running evaluation with inference threshold: {i} and training threshold: {t}')
		result = {'eval_threshold': i, 'train_threshold': t}

		# Make new output directory
		temp_dir = os.path.join(output_dir, f'threshold_i-p{str(i)[2:]}_t-p{str(t)[2:]}')
		if i in [result['eval_threshold'] for result in results] and t in [result['train_threshold'] for result in results]:
			print(f'Results already stored')
			continue

		# Load and save tokenizer
		temp_tokenizer = tokenizer
		if args.task == 'sst2' and args.model_name == 'bert-base' and not USE_NON_PRUNED:
			temp_tokenizer = tokenizer_wp
		temp_tokenizer.save_pretrained(temp_dir)

		# Initialize and save given model
		temp_model = model
		if args.task == 'sst2' and args.model_name == 'bert-base' and not USE_NON_PRUNED:
			temp_model = model_wp
		temp_model.save_pretrained(temp_dir)

		# Load and save new config
		config = BertConfig.from_pretrained(temp_dir)
		config.pruning_threshold = i
		config.k = k
		config.sparsity_file = os.path.join(temp_dir, 'sparsity.json')
		config.save_pretrained(temp_dir)

		if os.path.exists(config.sparsity_file): os.remove(config.sparsity_file)

		# Load model and prune weights
		if args.prune_weights: 
			# Do weight pruning with fixed threshold
			config.pruning_threshold = 0.025
			config.save_pretrained(temp_dir)

			if args.task == 'sst2':
				temp_model = DTBertForSequenceClassification.from_pretrained(temp_dir)
			else:
				temp_model = DTBertForQuestionAnswering.from_pretrained(temp_dir)

			if config.pruning_threshold > 0:
				temp_model.prune_weights()

				sparsity = json.load(open(config.sparsity_file))
				matrix_sizes, num_zeros = 0, 0
				for sp in sparsity:
					num_zeros += sp[0]
					matrix_sizes += sp[1]

				print(f'Resultant weight sparsity: {num_zeros / matrix_sizes : 0.03f}')
				result['weight_sparsity'] = num_zeros / matrix_sizes
			else:
				result['weight_sparsity'] = 0

		# Save (weight-pruned) model
		temp_model.save_pretrained(temp_dir)

		# Update config
		config.pruning_threshold = i
		config.save_pretrained(temp_dir)

		# Load model
		if args.task == 'sst2':
			temp_model = DTBertForSequenceClassification.from_pretrained(temp_dir)
		else:
			temp_model = DTBertForQuestionAnswering.from_pretrained(temp_dir)

		if os.path.exists(config.sparsity_file): 
			os.rename(config.sparsity_file, os.path.join(temp_dir, 'weight_sparsity.json'))

		# Run evaluation on the SST-2 task or the SQuAD task
		training_args = get_training_args(temp_dir, args.task, args.do_train, t, os.path.join(temp_dir, 'grad_sparsity.json'))
		start_time = time.time()
		metrics = run_glue(training_args) if args.task == 'sst2' else run_squad_legacy.evaluate(training_args, temp_model, tokenizer)
		end_time = time.time()
		print(metrics)

		if i > 0:
			sparsity = json.load(open(config.sparsity_file))
			matrix_sizes, num_zeros = 0, 0
			for sp in sparsity:
				num_zeros += sp[0]
				matrix_sizes += sp[1]

			print(f'Resultant activation sparsity: {num_zeros / matrix_sizes : 0.03f}')
			result['activation_sparsity'] = num_zeros / matrix_sizes
		else:
			result['activation_sparsity'] = 0
		if t > 0:
			sparsity = json.load(open(os.path.join(temp_dir, 'grad_sparsity.json')))
			matrix_sizes, num_zeros = 0, 0
			for sp in sparsity:
				num_zeros += sp[0]
				matrix_sizes += sp[1]

			print(f'Resultant gradient sparsity: {num_zeros / matrix_sizes : 0.03f}')
			result['grad_sparsity'] = num_zeros / matrix_sizes
		else:
			result['grad_sparsity'] = 0

		if args.task == 'sst2':
			result['accuracy'] = metrics['eval_accuracy'] 
		else:
			result['f1'] = metrics['f1']

		result['time'] = end_time - start_time
		
		results.append(result)
		json.dump(results, open(os.path.join(output_dir, 'results.json'), 'w+'))

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_ylabel('Accuracy on SST-2 task (%)' if args.task == 'sst2' else 'F1 Score on SQuAD-v2', color='k')
	metric = 'accuracy' if args.task == 'sst2' else 'f1'
	mult = 100.0 if args.task == 'sst2' else 1.0
	ax1.plot([result['eval_threshold' if args.max_eval_threshold is not None else 'train_threshold'] for result in results], [result[metric] * mult for result in results], color='k')
	
	ax1.set_xlabel('DynaTran pruning threshold' if args.max_eval_threshold is not None else 'DynaProp pruning threshold')
	ax2.set_ylabel('Activation sparsity (%)' if args.max_eval_threshold is not None else 'Grad. sparsity (%)')
	ax2.tick_params(axis='y', labelcolor='tab:red')
	ax2.plot([result['eval_threshold' if args.max_eval_threshold is not None else 'train_threshold'] for result in results], [result['activation_sparsity' if args.max_eval_threshold is not None else 'grad_sparsity'] * 100 for result in results], color='tab:red')

	plt.savefig(os.path.join(output_dir, 'results.pdf'), bbox_inches='tight')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--task',
		metavar='',
		type=str,
		default='sst2',
		help='Task to test pruning')
	parser.add_argument('--model_name',
		metavar='',
		type=str,
		default='bert-tiny',
		help='BERT model')
	parser.add_argument('--max_eval_threshold',
		metavar='',
		type=float,
		default=None,
		help='maximum threshold for DynaTran')
	parser.add_argument('--max_train_threshold',
		metavar='',
		type=float,
		default=None,
		help='maximum threhold for DynaProp')
	parser.add_argument('--do_train',
		dest='do_train',
		action='store_true',
		help='to train the model using DynaProp')
	parser.add_argument('--prune_weights',
		dest='prune_weights',
		action='store_true',
		help='to prune weights of the model')
	parser.set_defaults(prune_weights=False)

	args = parser.parse_args()

	main(args)

