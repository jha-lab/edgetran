# Run Grow-and-Prune on the BERT architecture using the FlexiBERT framework

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings/utils')
sys.path.append('../../txf_design-space/flexibert')

import argparse
import re
import torch
import shlex
from dataclasses import dataclass, field
from typing import Optional
from roberta_pretraining import pretrain
import shutil
import random
import yaml
import json
import numpy as np
from six.moves import cPickle as pickle
from tqdm import tqdm
import gc

import graph_util
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
BERT_BASE_LOSS = # TODO: add this once BERT-Base is pre-trained

CKPT_PATH = '' # Path to the grow-and-prune checkpoint
PREFIX_CHECKPOINT_DIR = "checkpoint"

USE_GPU_EE = False # Use GPU-EE partition on della cluster


def worker(models_dir: str,
	model_dict: dict,
	model_hash: str,
	chosen_neighbor_hash: str,
	cluster: str,
	id: str):
	"""Worker to pre-train the given model
	
	Args:
	    config_file (str): path to the grow-and-prune configuration file
	    model_dict (dict): model dictionary of the given model
	    models_dir (str): path to the models directory
	    model_hash (str): hash of the given model
	    chosen_neighbor_hash (str): hash of the chosen neighbor
	    cluster (str): name of the cluster - "adroit", "tiger" or "della"
	    id (str): PU-NetID that is used to run slurm commands
	
	Returns:
	    job_id (str): Job ID for the slurm scheduler
	"""
	print(f'Training model with hash: {model_hash} \n\tand model dictionary:\n{model_dict}.')
	print(f'Transfering weights from neighbor with hash: {chosen_neighbor_hash}.')

	chosen_neighbor_path = os.path.join(models_dir, chosen_neighbor_hash)
	model_path = os.path.join(models_dir, model_hash)

	chosen_neighbor_model = BertModelModular.from_pretrained(chosen_neighbor_path)

	# Finding the latest checkpoint for chosen neighbor
	re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
	content = os.listdir(chosen_neighbor_path)
	checkpoints = [
	        path
	        for path in content
	        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(chosen_neighbor_path, path))
	    ]
	checkpoint_dir = max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))

	tokenizer = RobertaTokenizer.from_pretrained('../txf_design-space/roberta_tokenizer/')
	config_new = BertConfig(vocab_size = tokenizer.vocab_size)
	config_new.from_model_dict_hetero(model_dict)
	
	# Transfer weights from chosen neighbor to the current model
	model = BertModelModular(config_new)
	model.load_model_from_source(chosen_neighbor_model)

	# Setting up checkpoint for the current model
	if os.path.exists(model_path):
	    shutil.rmtree(model_path)
	shutil.copytree(os.path.join(chosen_neighbor_path, checkpoint_dir), os.path.join(model_path, checkpoint_dir))
	os.remove(os.path.join(output_dir_new, checkpoint_dir, 'optimizer.pt'))
	# os.remove(os.path.join(output_dir_new, checkpoint_dir, 'scheduler.pt'))
	model.save_pretrained(os.path.join(model_path, checkpoint_dir))

	# Save model dictionary
	json.dump(model_dict, open(os.path.join(models_dir, model_hash, 'model_dict.json'), 'w+'))

	args = [['--cluster', cluster]]

	if cluster == 'della' and USE_GPU_EE:
		slurm_stdout = subprocess.check_output('squeue', shell=True, text=True)
		if 'gpu-ee' not in slurm_stdout:
			args.extend(['--partition', 'gpu-ee'])

	args.extend(['--id', id])
	args.extend(['--model_dir', model_path])
	
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
	header = ['ACCEL HASH', 'JOB ID', 'TRAIN TYPE', 'START TIME', 'ELAPSED TIME', 'STATUS']

	rows = []
	for job in model_jobs:
		start_time, elapsed_time, status = get_job_info(job['job_id'])
		rows.append([job['accel_hash'], job['job_id'], job['train_type'], start_time, elapsed_time, status])

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
		
		txf_dataset[model_hash] = {'model_dict': model_dict, 'losses': losses}

		if loss < best_loss:
			best_loss = loss
			best_hash = model_hash

	json.dump(txf_dataset, open(txf_dataset_file, 'w+'))
	
	return best_loss, best_hash


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

	# TODO: Implement automated grow-and-prune
	

	print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()


