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


def _get_training_args(seed, max_steps, learning_rate, per_gpu_batch_size, output_dir, local_rank):
    a = "--seed {} \
    --do_train \
    --max_seq_length 512 \
    --per_gpu_train_batch_size {} \
    --gradient_accumulation_steps 4 \
    --max_steps {} \
    --adam_epsilon 1e-6 \
    --adam_beta2 0.98 \
    --learning_rate {} \
    --weight_decay 0.01 \
    --save_total_limit 2 \
    --warmup_steps 1000 \
    --lr_scheduler_type linear \
    --output_dir {} \
    --local_rank {} \
        ".format(seed, per_gpu_batch_size, max_steps, learning_rate, output_dir, local_rank)
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
	assert len(checkpoints) > 0, f'No checkpoint found to continue pre-training in the output directory: {args.output_dir}'
	checkpoint_dir = max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))

	curr_steps = int(checkpoint_dir.split('-')[1])
	max_steps = curr_steps + args.steps

	seed = 0
	training_args = _get_training_args(seed, max_steps, args.learning_rate, 16, args.output_dir, args.local_rank)

	# Get model dictionary from output directory
	model_dict = json.load(open(os.path.join(args.output_dir, 'model_dict.json'), 'r'))

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

