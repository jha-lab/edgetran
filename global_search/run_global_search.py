# Run global search for best-performing models on the FlexiBERT design space

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

import argparse
import torch
import shlex
import shutil
import yaml
import json
import numpy as np
from tqdm import tqdm
import gc
import tabulate
import subprocess
import time
import collections
import re

from utils import graph_util
from utils import print_util as pu
from utils import embedding_util

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


PREFIX_CHECKPOINT_DIR = "checkpoint"

USE_GPU_EE = True # Use GPU-EE partition on della cluster (False, True, or 'ONLY')

INIT_SAMPLER = 'Lhs' # Should be in ['Sobol', 'Lhs', 'Halton', Hammersly']
INIT_SAMPLES = 16 # Should be power of 2 and > 1

PRETRAIN_STEPS = 1000000 # Number of total steps for pre-training
LEARNING_RATE = 1e-4 # Learning rate for pre-training


def worker(models_dir: str,
	model_dict: dict,
	model_hash: str,
	steps: int,
	learning_rate: float,
	cluster: str,
	id: str):
	"""Worker to pre-train the given model
	
	Args:
		config_file (str): path to the grow-and-prune configuration file
		model_dict (dict): model dictionary of the given model
		models_dir (str): path to the models directory
		model_hash (str): hash of the given model
		chosen_neighbor_hash (str): hash of the chosen neighbor
		steps (int): number of steps for pre-training
		learning_rate (float): learning rate for pre-training
		config (dict): configuration for grow-and-prune
		cluster (str): name of the cluster - "adroit", "tiger" or "della"
		id (str): PU-NetID that is used to run slurm commands
	
	Returns:
		job_id (str): Job ID for the slurm scheduler
	"""
	print(f'{pu.bcolors.OKBLUE}Training model with hash:{pu.bcolors.ENDC}\n\t{model_hash} \n{pu.bcolors.OKBLUE}and model dictionary:{pu.bcolors.ENDC}\n\t{model_dict}.')

	model_path = os.path.join(models_dir, model_hash)

	tokenizer = RobertaTokenizer.from_pretrained('../txf_design-space/roberta_tokenizer/')
	config_new = BertConfig(vocab_size = tokenizer.vocab_size)
	config_new.from_model_dict_hetero(model_dict)
	
	# Initialize BERT model for pre-training
	model = BertForMaskedLMModular(config_new)

	# Save untrained model
	model.save_pretrained(model_path)

	# Save model dictionary
	json.dump(model_dict, open(os.path.join(model_path, 'model_dict.json'), 'w+'))

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
	args.extend(['--steps', str(steps)])
	args.extend(['--learning_rate', str(learning_rate)])
	
	slurm_stdout = subprocess.check_output(
		f'ssh della-gpu "cd /scratch/gpfs/stuli/edge_txf/global_search; source ./job_scripts/job_worker.sh {" ".join(args)}"',
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


def get_latest_checkpoint(model_hash: str, models_dir: str):
	"""Get latest trained checkpoint for the given model
	
	Args:
		model_hash (str): hash of the given model
		models_dir (str): path to the models directory

	Returns:
		checkpoint (int): latest trained checkpoint
	"""
	checkpoint = None

	model_path = os.path.join(models_dir, model_hash)

	if not os.path.exists(model_path):
		print(f'{pu.bcolors.WARNING}Could not find model path for the given hash: {model_hash}{pu.bcolors.ENDC}')
		return checkpoint

	# Finding the latest checkpoint for the given model
	re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
	content = os.listdir(model_path)
	checkpoints = [
			path
			for path in content
			if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(model_path, path))
		]
	if len(checkpoints) == 0: return checkpoint
	checkpoint_dir = max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))

	checkpoint = str(checkpoint_dir.split('-')[-1])

	return checkpoint


def is_model_in_queue(model_hash: str, model_jobs: list):
    """To check if the model is still being pre-trained
    
    Args:
        model_hash (str): hash for the given model
        model_jobs (list): list of jobs
    """
    job_ids = []
    for job in model_jobs:
    	if job['model_hash'] == model_hash: job_ids.append(job['job_id'])

    if len(job_ids) == 0: return False

    if get_job_info(job_ids[-1])[2] == 'RUNNING' or get_job_info(job_ids[-1])[2] == 'PENDING': return True

    return False


def print_jobs(model_jobs: list, models_dir: str):
	"""Print summary of all completed, pending and running jobs
	
	Args:
		model_jobs (list): list of jobs
		models_dir (str): path to the models directory
	"""
	header = ['MODEL HASH', 'JOB ID', 'START TIME', 'ELAPSED TIME', 'STATUS', 'LATEST CKPT']

	rows = []
	for job in model_jobs:
		start_time, elapsed_time, status = get_job_info(job['job_id'])
		rows.append([job['model_hash'], job['job_id'], start_time, elapsed_time, status, get_latest_checkpoint(job['model_hash'], models_dir)])

	print()
	print(tabulate.tabulate(rows, header))


def wait_for_jobs(model_jobs: list, models_dir: str, running_limit: int = 4, patience: int = 1):
	"""Wait for current jobs in queue to complete
	
	Args:
		model_jobs (list): list of jobs
		models_dir (str): path to the models directory
		running_limit (int, optional): number of running jobs to limit
		patience (int, optional): number of pending jobs to wait for
	"""
	print_jobs(model_jobs, models_dir)

	completed_jobs = 0
	last_completed_jobs = 0
	last_pending_jobs = 0
	running_jobs = np.inf
	pending_jobs = np.inf
	while running_jobs > running_limit or pending_jobs > patience:
		completed_jobs, running_jobs, pending_jobs = 0, 0, 0
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED': 
				completed_jobs += 1
			elif status == 'PENDING' or status == 'UNKNOWN':
				pending_jobs += 1
			elif status == 'RUNNING':
				running_jobs += 1
			elif status == 'FAILED':
				print_jobs(model_jobs, models_dir)
				# print(f'{pu.bcolors.FAIL}Some jobs failed{pu.bcolors.ENDC}')
				raise RuntimeError('Some jobs failed.')
		if last_completed_jobs != completed_jobs or last_pending_jobs != pending_jobs:
			print_jobs(model_jobs, models_dir)
		last_completed_jobs = completed_jobs 
		last_pending_jobs = pending_jobs
		time.sleep(10)


def main():
	"""Run BOSHCODE to get the best CNN-Accelerator pair in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--design_space_file',
		metavar='',
		type=str,
		help='path to the design space configuration file',
		default='./design_space/design_space.yaml')
	parser.add_argument('--txf_dataset_file',
		metavar='',
		type=str,
		help='path to the transformer dataset file',
		default='./dataset/dataset.json')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to the directory where all models are trained',
		default='../models/global_search/')
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

	# Load design space to run global search on
	design_space = yaml.safe_load(open(args.design_space_file))

	# Load dataset file if previously generated
	if os.path.exists(args.txf_dataset_file):
		dataset = json.load(open(args.txf_dataset_file))
		for key in dataset.keys():
			dataset[key]['embedding'] = eval(dataset[key]['embedding'])
		print(f'{pu.bcolors.OKGREEN}Loaded dataset from: {args.txf_dataset_file}{pu.bcolors.ENDC}')
	else:
		# Generate samples
		dataset = embedding_util.get_samples(design_space, num_samples=INIT_SAMPLES, sampling_method=INIT_SAMPLER, debug=True)

		# Save dataset
		json_dataset = {}
		for key, value in dataset.items():
			json_dataset[key] = {'model_dict': value['model_dict'], 'model_type': value['model_type'], 'embedding': str(value['embedding'])}

		json.dump(json_dataset, open(args.txf_dataset_file, 'w+'))
		print(f'{pu.bcolors.OKGREEN}Saved dataset with {len(dataset)} models to: {args.txf_dataset_file}{pu.bcolors.ENDC}')

	# Instantiate list of jobs
	model_jobs = json.load(open('./dataset/model_jobs.json', 'r')) if os.path.exists('./dataset/model_jobs.json') else []

	# If jobs in queue, wait for those jobs to complete first
	if len(model_jobs) > 0: wait_for_jobs(model_jobs, args.models_dir, patience=0)

	all_trained = False
	while all_trained == False:
		# Run pre-training jobs
		jobs_added = False
		for model_hash in dataset.keys():
			if get_latest_checkpoint(model_hash, args.models_dir) != str(PRETRAIN_STEPS) and not is_model_in_queue(model_hash, model_jobs):
				# Send model to worker for pre-training
				job_id = worker(args.models_dir, dataset[model_hash]['model_dict'], model_hash, PRETRAIN_STEPS, LEARNING_RATE, args.cluster, args.id)

				# Add job to model_jobs
				model_jobs.append({'model_hash': model_hash, 'job_id': job_id})
				json.dump(model_jobs, open('./dataset/model_jobs.json', 'w+'))

				jobs_added = True

		# Wait for jobs to be completed
		if jobs_added: wait_for_jobs(model_jobs, args.models_dir)

		# Some models need to be run again if their jobs did not finish training
		all_trained = True
		for model_hash in dataset.keys():
			if get_latest_checkpoint(model_hash, args.models_dir) != str(PRETRAIN_STEPS):
				all_trained = False
				break

	print(f'{pu.bcolors.OKGREEN}All initial samples have been pre-trained!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()
