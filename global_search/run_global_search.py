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
INIT_SAMPLES = 16 # Should be power of 2


def worker(models_dir: str,
	model_dict: dict,
	model_hash: str,
	chosen_neighbor_hash: str,
	steps: int,
	learning_rate: float,
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
		steps (int): number of steps for pre-training
		learning_rate (float): learning rate for pre-training
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
	wt_ratio = model.load_model_from_source(chosen_neighbor_model, debug=True)

	print(f'Weight transfer ratio: {wt_ratio}')

	# Setting up checkpoint for the current model
	if os.path.exists(model_path):
		shutil.rmtree(model_path)
	shutil.copytree(os.path.join(chosen_neighbor_path), os.path.join(model_path))
	try:
		os.remove(os.path.join(model_path, checkpoint_dir, 'scheduler.pt'))
		os.remove(os.path.join(model_path, checkpoint_dir, 'optimizer.pt'))
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
	args.extend(['--steps', str(steps)])
	args.extend(['--learning_rate', str(learning_rate)])
	
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
			elif status == 'PENDING' or status == 'UNKNOWN':
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
	model_jobs = []


if __name__ == '__main__':
	main()
