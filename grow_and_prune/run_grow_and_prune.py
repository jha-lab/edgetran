# Run Grow-and-Prune on the BERT architecture using the FlexiBERT framework

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

import argparse
import re
import torch
import shlex
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
from roberta_pretraining import pretrain
import shutil
import random
import yaml
import json
import numpy as np
from six.moves import cPickle as pickle
from tqdm import tqdm
import gc
import tabulate
import subprocess
import time

from utils import graph_util
from utils import print_util as pu

from datasets import load_dataset, interleave_datasets, load_from_disk
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular
from transformers import BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers import (
	CONFIG_MAPPING,
	MODEL_FOR_MASKED_LM_MAPPING,
	AutoConfig,
	AutoModelForMaskedLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	HfArgumentParser,
	Trainer,
	TrainingArguments,
	set_seed,
)


BERT_BASE_HASH = '07aaba14d29455a984e2aef6312a8870'
BERT_BASE_LOSS = 1.2 # TODO: update this once BERT-Base is pre-trained

CKPT_PATH = '' # Path to the grow-and-prune checkpoint
PREFIX_CHECKPOINT_DIR = "checkpoint"

USE_GPU_EE = 'ONLY' # Use GPU-EE partition on della cluster (False, True, or 'ONLY')

PERFORMANCE_PATIENCE = 5
PRETRAIN_STEPS = 10000 # Steps to pre-train beyond the latest checkpoint


def worker(models_dir: str,
	model_dict: dict,
	model_hash: str,
	chosen_neighbor_hash: str,
	config: dict,
	cluster: str,
	id: str):
	"""Worker to pre-train the given model
	
	Args:
		config_file (str): path to the grow-and-prune configuration file
		model_dict (dict): model dictionary of the given model
		models_dir (str): path to the models directory
		model_hash (str): hash of the given model
		chosen_neighbor_hash (str): hash of the chosen neighbor
		config (dict): configuration for grow-and-prune
		cluster (str): name of the cluster - "adroit", "tiger" or "della"
		id (str): PU-NetID that is used to run slurm commands
	
	Returns:
		job_id (str): Job ID for the slurm scheduler
	"""
	print(f'Training model with hash:\n\t{model_hash} \nand model dictionary:\n\t{model_dict}.')
	print(f'Transfering weights from neighbor with hash: {chosen_neighbor_hash}.')

	chosen_neighbor_path = os.path.join(models_dir, chosen_neighbor_hash)
	model_path = os.path.join(models_dir, model_hash)

	chosen_neighbor_model = BertForMaskedLMModular.from_pretrained(chosen_neighbor_path)

	# Finding the latest checkpoint for chosen neighbor
	re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
	content = os.listdir(chosen_neighbor_path)
	checkpoints = [
			path
			for path in content
			if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(chosen_neighbor_path, path))
		]
	checkpoint_dir = max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))

	tokenizer = RobertaTokenizer.from_pretrained('../txf_design-space/roberta_tokenizer/')
	config_new = BertConfig(vocab_size = tokenizer.vocab_size)
	config_new.from_model_dict_hetero(model_dict)
	
	# Transfer weights from chosen neighbor to the current model
	model = BertForMaskedLMModular(config_new, transfer_mode=config['model_transfer_mode'])
	wt_ratio = model.load_model_from_source(chosen_neighbor_model)

	print(f'Weight transfer ratio: {wt_ratio}')

	# Setting up checkpoint for the current model
	if os.path.exists(model_path):
		shutil.rmtree(model_path)
	shutil.copytree(os.path.join(chosen_neighbor_path), os.path.join(model_path))
	try:
		os.remove(os.path.join(model_path, checkpoint_dir, 'optimizer.pt'))
		# os.remove(os.path.join(model_path, checkpoint_dir, 'scheduler.pt'))
	except:
		pass
	model.save_pretrained(os.path.join(model_path, checkpoint_dir))

	# Save model dictionary
	json.dump(model_dict, open(os.path.join(models_dir, model_hash, 'model_dict.json'), 'w+'))

	args = ['--cluster', cluster]

	if cluster == 'della':
		if USE_GPU_EE is True:
			slurm_stdout = subprocess.check_output('squeue', shell=True, text=True)
			if 'gpu-ee' not in slurm_stdout:
				args.extend(['--partition', 'gpu-ee'])
		elif USE_GPU_EE == 'ONLY':
			args.extend(['--partition', 'gpu-ee'])

	args.extend(['--id', id])
	args.extend(['--model_hash', model_hash])
	args.extend(['--model_dir', model_path])
	args.extend(['--steps', str(PRETRAIN_STEPS)])
	
	slurm_stdout = subprocess.check_output(
		f'ssh della-gpu "cd /scratch/gpfs/stuli/edge_txf/grow_and_prune; source ./job_scripts/job_worker.sh {" ".join(args)}"',
		shell=True, text=True, executable="/bin/bash")

	return slurm_stdout.split()[-1]
		

def get_job_info(job_id: int):
	"""Obtain job info
	
	Args:
		job_id (int): job id
	
	Returns:
		start_time, elapsed_time, status (str, str, str): job details
	"""
	slurm_stdout = subprocess.check_output(f'ssh della-gpu "slist {job_id}"', shell=True, text=True, executable="/bin/bash")
	slurm_stdout = slurm_stdout.split('\n')[2].split()

	if len(slurm_stdout) > 7:
		start_time, elapsed_time, status = slurm_stdout[5], slurm_stdout[6], slurm_stdout[7]
		if start_time == 'Unknown': start_time = 'UNKNOWN'
	else:
		start_time, elapsed_time, status = 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

	return start_time, elapsed_time, status


def print_jobs(model_jobs: list):
	"""Print summary of all completed, pending and running jobs
	
	Args:
		model_jobs (list): list of jobs
	"""
	header = ['MODEL HASH', 'JOB ID', 'START TIME', 'ELAPSED TIME', 'STATUS']

	rows = []
	for job in model_jobs:
		start_time, elapsed_time, status = get_job_info(job['job_id'])
		rows.append([job['model_hash'], job['job_id'], start_time, elapsed_time, status])

	print()
	print(tabulate.tabulate(rows, header))


def wait_for_jobs(model_jobs: list, txf_dataset: dict, running_limit: int = 4, patience: int = 1):
	"""Wait for current jobs in queue to complete
	
	Args:
		model_jobs (list): list of jobs
		txf_dataset (dict): dictionary of transformers and their losses
		running_limit (int, optional): number of running jobs to limit
		patience (int, optional): number of pending jobs to wait for
	"""
	print_jobs(model_jobs)

	completed_jobs = 0
	last_completed_jobs = 0
	running_jobs = np.inf
	pending_jobs = np.inf
	while running_jobs > running_limit or pending_jobs > patience:
		completed_jobs, running_jobs, pending_jobs = 0, 0, 0
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED': 
				completed_jobs += 1
			elif status == 'PENDING':
				pending_jobs += 1
			elif status == 'RUNNING':
				running_jobs += 1
			elif status == 'FAILED':
				print_jobs(model_jobs)
				print(f'{pu.bcolors.FAIL}Some jobs failed{pu.bcolors.ENDC}')
				# raise RuntimeError('Some jobs failed.')
		if last_completed_jobs != completed_jobs:
			print_jobs(model_jobs)
		last_completed_jobs = completed_jobs 
		time.sleep(1)


def update_dataset(txf_dataset: dict,
	models_dir: str, 
	txf_dataset_file: str,
	save_dataset=True):
	"""Update the dataset with all pre-trained models
	
	Args:
		txf_dataset (dict): dictionary of transformers and their losses
		models_dir (str): path to the models directory
		txf_dataset_file (str): path to the transformers dataset file
	
	Returns:
		best_loss, best_hash (float, str): lowest loss (and corresponding hash) among all transformers trained
	"""
	best_loss, best_hash = np.inf, ''
	for model_hash in os.listdir(models_dir):
		log_history = json.load(open(os.path.join(models_dir, model_hash, 'log_history.json'), 'r'))
		losses = [state['loss'] for state in log_history[:-1]]
		model_dict = json.load(open(os.path.join(models_dir, model_hash, 'model_dict.json'), 'r'))
		
		txf_dataset[model_hash] = {'losses': losses}

		if losses[-1] < best_loss:
			best_loss = losses[-1]
			best_hash = model_hash

	json.dump(txf_dataset, open(txf_dataset_file, 'w+'))

	print(f'Model with best loss ({best_loss}) has hash: {best_hash}')
	
	return best_loss, best_hash


def get_attention_weights(models_dir: str,
	model_hash: str):
	"""Obtain the weights of the attention heads
	
	Args:
		models_dir (str): path to the models directory\
		model_hash (str): hash of the given model
	
	Returns:
		attention_weights (list): mean weight of the attention heads
	"""

	# Get model dictionary
	model_dict = json.load(open(os.path.join(models_dir, model_hash, 'model_dict.json'), 'r'))

	# Finding the latest checkpoint for chosen model
	re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
	content = os.listdir(os.path.join(models_dir, model_hash))
	checkpoints = [
			path
			for path in content
			if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(models_dir, model_hash, path))
		]
	checkpoint_dir = max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))

	# Initialize FlexiBERT model
	model = BertForMaskedLMModular.from_pretrained(os.path.join(models_dir, model_hash, checkpoint_dir))

	# Instantiate attention head weights
	attention_weights = []

	for i in range(model_dict['l']):
		# Get multi-head attention for current encoder layer
		attention_head_size = int(model_dict['o'][i][0].split('_')[2])
		attention_layer = model.bert.encoder.layer[i].attention.self

		wma_count, conv_count = 0, 0
		for j in range(len(model_dict['o'][i])):
			# Get mean weight values for each attention head
			query_mean = torch.mean(torch.square(
				attention_layer.query.weight[j*attention_head_size:(j+1)*attention_head_size])).item()
			key_mean = torch.mean(torch.square(
				attention_layer.key.weight[j*attention_head_size:(j+1)*attention_head_size])).item()
			value_mean = torch.mean(torch.square(
				attention_layer.value.weight[j*attention_head_size:(j+1)*attention_head_size])).item()
			weights = [query_mean, key_mean, value_mean]

			# Add more weights based on attention operation
			if model_dict['o'][i][j].split('_')[1] == 'wma':
				wma_mean = torch.mean(torch.square(
					getattr(attention_layer, f'W{wma_count}'))).item()
				weights.append(wma_mean)
				wma_count += 1
			elif model_dict['o'][i][j].split('_')[1].isnumeric():
				key_conv_attn_layer_mean = np.mean([torch.mean(torch.square(
					getattr(attention_layer, f'key_conv_attn_layer{conv_count}').depthwise.weight)).item(),
													torch.mean(torch.square(
					getattr(attention_layer, f'key_conv_attn_layer{conv_count}').pointwise.weight)).item()])
				conv_kernel_layer_mean = torch.mean(torch.square(
					getattr(attention_layer, f'conv_kernel_layer{conv_count}').weight)).item()
				conv_out_layer_mean = torch.mean(torch.square(
					getattr(attention_layer, f'conv_out_layer{conv_count}').weight)).item()
				conv_mean = np.mean([key_conv_attn_layer_mean, conv_kernel_layer_mean, conv_out_layer_mean])
				weights.append(conv_mean)
				conv_count += 1

			# print(f'Layer {i}, Attention head {j}, mean: {np.mean(weights): 0.3e}')
			attention_weights.append({'layer': i, 'attention_head': j, 'mean_weight': np.mean(weights)})

	# print(attention_weights)
	return attention_weights


def get_feed_forward_weights(models_dir: str,
	model_hash: str):
	"""Obtain the weights of the feed-forward layers 

	Args:
		models_dir (str): path to the models directory
		model_hash (str): hash of the given model
	
	Returns:
		feed_forward_weights (list): mean weight of the feed-forward layers
	"""

	# Get model dictionary
	model_dict = json.load(open(os.path.join(models_dir, model_hash, 'model_dict.json'), 'r'))

	# Finding the latest checkpoint for chosen model
	re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
	content = os.listdir(os.path.join(models_dir, model_hash))
	checkpoints = [
			path
			for path in content
			if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(models_dir, model_hash, path))
		]
	checkpoint_dir = max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))

	# Initialize FlexiBERT model
	model = BertForMaskedLMModular.from_pretrained(os.path.join(models_dir, model_hash, checkpoint_dir))

	# Instantiate feed-forward weights
	feed_forward_weights = []

	# Get feed-forward weights for pruning
	for i in range(model_dict['l']):
		for j in range(len(model_dict['f'][i])):
			with torch.no_grad():
				mean_weight = np.mean(np.abs(model.bert.encoder.layer[i].intermediate.sequential[j].weight.cpu().numpy()))
			
			feed_forward_weights.append({'layer': i, 'feed_forward_layer': j, 'mean_weight': mean_weight})

	# print(feed_forward_weights)
	return feed_forward_weights


def main():
	"""Run BOSHCODE to get the best CNN-Accelerator pair in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config_file',
		metavar='',
		type=str,
		help='path to the grow-and-prune configuration file',
		default='./configs/config.yaml')
	parser.add_argument('--txf_dataset_file',
		metavar='',
		type=str,
		help='path to the transformer dataset file',
		default='./dataset/dataset_base.json')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to the directory where all models are trained',
		default='../models')
	parser.add_argument('--n_jobs',
		metavar='',
		type=int,
		help='number of parallel jobs for training the transformers',
		default=8)
	parser.add_argument('--cluster',
		metavar='',
		type=str,
		help='name of the cluster - "adroit", "tiger" or "della"',
		default='della')
	parser.add_argument('--id',
		metavar='',
		type=str,
		help='PU-NetID that is used to run slurm commands',
		default='stuli')

	args = parser.parse_args()

	random_seed = 0

	# Load configurations for grow-and-prune
	config = yaml.safe_load(open(args.config_file))

	# Starts with BERT-Base
	best_loss = BERT_BASE_LOSS
	best_hash = BERT_BASE_HASH

	# Get transformer dataset
	txf_dataset = {}
	if os.path.exists(args.txf_dataset_file):
		txf_dataset = json.load(open(args.txf_dataset_file, 'r'))

	best_loss, best_hash = update_dataset(txf_dataset, args.models_dir, args.txf_dataset_file)
	best_model_dict = json.load(open(os.path.join(args.models_dir, best_hash, 'model_dict.json'), 'r'))

	old_best_loss = best_loss

	# If this script is run for the first time, the best model is BERT-Base
	assert best_hash == BERT_BASE_HASH, 'If script is run for the first time, best model should be BERT-Base'

	# Instantiate list of jobs
	model_jobs = []

	same_performance = 0
	while same_performance < PERFORMANCE_PATIENCE:
		best_loss, best_hash = update_dataset(txf_dataset, args.models_dir, args.txf_dataset_file)
		best_model_dict = json.load(open(os.path.join(args.models_dir, best_hash, 'model_dict.json'), 'r'))

		# Prune model based on configuration
		if best_hash != BERT_BASE_HASH:
			model_dict = deepcopy(best_model_dict)

			# Get attention weights for the model
			attention_weights = get_attention_weights(args.models_dir, best_hash)

			# Sort attention heads based on increasing mean weight
			attention_weights.sort(key=lambda x:x['mean_weight'])
			attention_weights_unsorted = deepcopy(attention_weights)

			# Prune attention heads
			for num_op in range(config['prune']['num_ops']):
				# Remove attention head with lowest mean weight values
				model_dict['o'][attention_weights[num_op]['layer']].pop(attention_weights[num_op]['attention_head'])

			# Get feed-forward weights for the model
			feed_forward_weights = get_feed_forward_weights(args.models_dir, best_hash)

			# Sort feed-forward based on increasing mean weight
			feed_forward_weights.sort(key=lambda x:x['mean_weight'])
			feed_forward_weights_unsorted = deepcopy(feed_forward_weights)

			# Prune feed-forward layers
			for num_ff_layer in range(config['prune']['num_feed_forward_layers']):
				model_dict['f'][feed_forward_weights[num_ff_layer]['layer']][feed_forward_weights[num_ff_layer]['feed_forward_layer']] -= \
					config['prune']['feed_forward_prune_dim']

			# Prune encoder layer:
			min_encoder_weight, min_encoder_idx = np.inf, 0
			for i in range(model_dict['l']):
				mean_encoder_weight = np.mean([attention_weights_unsorted[i]['mean_weight'], feed_forward_weights_unsorted[i]['mean_weight']])
				if mean_encoder_weight < min_encoder_weight:
					min_encoder_weight = mean_encoder_weight
					min_encoder_idx = i
			model_dict['h'][min_encoder_idx] -= config['prune']['hidden_prune_dim']
			for j in range(len(model_dict['o'][min_encoder_idx])):
				model_dict['o'][min_encoder_idx][j] = model_dict['o'][min_encoder_idx][j].split('_')[0] + '_' + \
					model_dict['o'][min_encoder_idx][j].split('_')[1] + '_' + str(int(model_dict['h'][min_encoder_idx]/12))
			
			# Get the hash of the current model
			model_graph = graph_util.model_dict_to_graph(model_dict)
			model_hash = graph_util.hash_graph(*model_graph)

			# Train pruned model
			print(f'Training pruned model wih dictionary:\n\t{model_dict}\nand hash:\n\t{model_hash}')
			job_id = worker(args.models_dir, model_dict, model_hash, 
				chosen_neighbor_hash=best_hash, config=config, cluster=args.cluster, id=args.id)

			model_jobs.append({'model_hash': model_hash, 'job_id': job_id})

			# Wait for jobs to complete
			wait_for_jobs(model_jobs, txf_dataset, running_limit=0, patience=0)

			best_loss, best_hash = update_dataset(txf_dataset, args.models_dir, args.txf_dataset_file)
			best_model_dict = json.load(open(os.path.join(args.models_dir, best_hash, 'model_dict.json'), 'r'))

			assert best_hash == model_hash, f'Pruned model (with hash: {model_hash}) does not give the best loss (best model with hash: {best_hash})'

		# Grow current best model based on configuration
		for i in range(config['num_grow_samples']):
			model_dict = deepcopy(best_model_dict)

			# Add a random attention operation
			for num_op in range(config['grow']['num_ops']):
				layer = random.randint(0, model_dict['l']-1)
				op = random.sample(config['allowed_ops'], 1)[0]
				
				layer_hidden_dim = model_dict['o'][layer][0].split('_')[2]
				model_dict['o'][layer].append(op + '_' +  layer_hidden_dim)
				
			# Add a feed-forward stack
			layer = random.randint(0, model_dict['l']-1)
			model_dict['f'][layer].append(model_dict['f'][layer][-1])

			# Get the hash of the current model
			model_graph = graph_util.model_dict_to_graph(model_dict)
			model_hash = graph_util.hash_graph(*model_graph)

			# Train sampled model
			job_id = worker(args.models_dir, model_dict, model_hash, 
				chosen_neighbor_hash=best_hash, config=config, cluster=args.cluster, id=args.id)

			model_jobs.append({'model_hash': model_hash, 'job_id': job_id})

		# Wait for jobs to complete
		wait_for_jobs(model_jobs, txf_dataset, running_limit=0, patience=0)

		# Update same_performance to check convergence
		if best_loss == old_best_loss:
			same_performance += 1
		old_best_loss = best_loss

		break	

	print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()


