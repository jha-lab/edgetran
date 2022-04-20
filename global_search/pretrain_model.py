# Run pre-training for the given FlexiBERT model

# Author : Shikhar Tuli

import os
import sys

sys.path.append('../txf_design-space/transformers/src/')
sys.path.append('../txf_design-space/embeddings/')
sys.path.append('../txf_design-space/flexibert/')

import os
import torch

import shlex
from utils import graph_util
import argparse
import re
import json

from roberta_pretraining import pretrain
from matplotlib import pyplot as plt

import logging
#logging.disable(logging.INFO)
#logging.disable(logging.WARNING)


PREFIX_CHECKPOINT_DIR = "checkpoint"


def _get_training_args(seed, overwrite_ouput_dir, max_steps, learning_rate, per_gpu_batch_size, gradient_accumulation_steps, output_dir, local_rank):
    if overwrite_ouput_dir:
    	a = "--seed {} \
		    --do_train \
		    --max_seq_length 512 \
		    --per_device_train_batch_size {} \
		    --max_steps {} \
		    --adam_epsilon 1e-6 \
		    --adam_beta2 0.99 \
		    --learning_rate {} \
		    --weight_decay 0.01 \
		    --save_total_limit 2 \
		    --warmup_steps 10000 \
		    --lr_scheduler_type linear \
		    --gradient_accumulation_steps {} \
		    --overwrite_output_dir \
		    --output_dir {} \
		    --local_rank {} \
		        ".format(seed, per_gpu_batch_size, max_steps, learning_rate, gradient_accumulation_steps, output_dir, local_rank)
    else:
    	a = "--seed {} \
		    --do_train \
		    --max_seq_length 512 \
		    --per_device_train_batch_size {} \
		    --max_steps {} \
		    --adam_epsilon 1e-6 \
		    --adam_beta2 0.99 \
		    --learning_rate {} \
		    --weight_decay 0.01 \
		    --save_total_limit 2 \
		    --warmup_steps 10000 \
		    --lr_scheduler_type linear \
		    --gradient_accumulation_steps {} \
		    --output_dir {} \
		    --local_rank {} \
		        ".format(seed, per_gpu_batch_size, max_steps, learning_rate, gradient_accumulation_steps, output_dir, local_rank)
    return shlex.split(a)


def main(args):
	"""Pretraining front-end function"""

	# Finding the latest checkpoint for chosen model
	re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
	content = os.listdir(args.output_dir)
	checkpoints = [
			path
			for path in content
			if re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(args.output_dir, path))
		]
	overwrite_output_dir = False if len(checkpoints) > 0 else True

	# Get model dictionary from output directory
	model_dict = json.load(open(os.path.join(args.output_dir, 'model_dict.json'), 'r'))

	# Set initial training parameters
	seed = 0
	if args.trial_run:
		batch_size, gradient_accumulation_steps = 64, 1 # Here we assume pre-training to be run on 4 GPUs, i.e., a net batch size of 256
	else:
		batch_size_file = json.load(open(os.path.join(args.output_dir, 'batch_size.json'), 'r'))
		batch_size, gradient_accumulation_steps = batch_size_file['batch_size'], batch_size_file['gradient_accumulation_steps']

	if args.trial_run:
		# Run pre-training, reduce batch size if it fails
		metrics, log_history, model = None, None, None
		while batch_size >= 1:
			try:
				training_args = _get_training_args(seed, overwrite_output_dir, 1, args.learning_rate, batch_size, gradient_accumulation_steps, args.output_dir, args.local_rank)

				# Pre-train model
				metrics, log_history, model = pretrain(training_args, model_dict)
			except Exception as e:
				print(f'Encountered error: \n{e} \nReducing batch size to {batch_size//2}...')
				batch_size, gradient_accumulation_steps = batch_size//2, 2 * gradient_accumulation_steps
			else:
				break

		if batch_size < 1: raise RuntimeError(f'Batch size reached below 1. Can\'t fit model.')

		# Store batch size and gradient accumulation steps for final pre-training
		json.dump({'batch_size': batch_size, 'gradient_accumulation_steps': gradient_accumulation_steps}, open(os.path.join(args.output_dir, 'batch_size.json'), 'w+'))
	else:
		training_args = _get_training_args(seed, overwrite_output_dir, args.steps, args.learning_rate, batch_size, gradient_accumulation_steps, args.output_dir, args.local_rank)

		# Pre-train model
		metrics, log_history, model = pretrain(training_args, model_dict)

	# Save log history
	json.dump(log_history, open(os.path.join(args.output_dir, 'log_history.json'), 'w+'))

	# Plot and save log history
	plt.plot([state['step'] for state in log_history[:-1]], [state['loss'] for state in log_history[:-1]])
	plt.xlabel('Training steps')
	plt.ylabel('Loss')
	plt.savefig(os.path.join(args.output_dir, 'loss.pdf'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--output_dir',
		metavar='',
		type=str,
		help='path to save the pretrained model')
	parser.add_argument('--trial_run',
		action='store_true',
		help='trial run')
	parser.add_argument('--steps',
		metavar='',
		type=int,
		help='number of steps to pre-train beyond latest checkpoint')
	parser.add_argument('--learning_rate',
		metavar='',
		type=float,
		help='learning rate to set for training')
	parser.add_argument('--local_rank',
		metavar='',
		type=int,
		help='rank of the process during distributed training',
		default=-1)

	args = parser.parse_args()

	main(args)

