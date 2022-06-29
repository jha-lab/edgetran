# Run pre-training for the given FlexiBERT model

# Author : Shikhar Tuli

import os
import sys

sys.path.append('../txf_design-space/transformers/src/')
sys.path.append('../txf_design-space/embeddings/')
sys.path.append('../txf_design-space/flexibert/')

import torch
import shlex
from utils import graph_util
from utils.run_glue import main as run_glue
import argparse
import re
import json

from roberta_pretraining import pretrain
from load_all_glue_datasets import main as load_all_glue_datasets
from datasets import load_dataset, load_metric
from tokenize_glue_datasets import save_dataset
from finetune_flexibert import finetune
from matplotlib import pyplot as plt

sys.path.append('../txf_design-space/transformers/src/transformers')
from transformers import BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular, BertForSequenceClassificationModular

import logging
#logging.disable(logging.INFO)
#logging.disable(logging.WARNING)


PREFIX_CHECKPOINT_DIR = "checkpoint"
GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']


def get_training_args(pretrained_dir, output_dir, task, autotune, autotune_trials):

	training_args = f'--model_name_or_path {pretrained_dir} \
		--task_name {task} \
		--do_train \
		--do_eval \
		{"--autotune" if autotune else ""} \
		--autotune_trials {autotune_trials} \
		--logging_steps 50 \
		--max_seq_length 512 \
		--per_device_train_batch_size 16 \
		--gradient_accumulation_steps 4 \
		--load_best_model_at_end \
		--metric_for_best_model eval_loss \
		--learning_rate 2e-5 \
		--weight_decay 0.01 \
		--num_train_epochs 5 \
		--overwrite_output_dir \
		--output_dir {output_dir}'

	training_args = shlex.split(training_args)

	return training_args


def get_tokenizer_args(output_dir, task):

	training_args = f'--task_name {task} \
		--do_train \
		--do_eval \
		--max_seq_length 512 \
		--output_dir {output_dir}\
		--overwrite_output_dir'

	training_args = shlex.split(training_args)

	return training_args


def main(args):
	"""Finetuning front-end function"""

	output_dir = os.path.join(args.pretrained_dir, 'glue')

	model_dict = json.load(open(os.path.join(args.pretrained_dir, 'model_dict.json'), 'r'))

	# Load all GLUE datasets
	load_all_glue_datasets()

	# Load tokenizer and get model configuration
	tokenizer = RobertaTokenizer.from_pretrained('../txf_design-space/roberta_tokenizer/')
	tokenizer.save_pretrained(output_dir)

	config_new = BertConfig(vocab_size=tokenizer.vocab_size)
	config_new.from_model_dict_hetero(model_dict)
	config_new.save_pretrained(output_dir)
	
	# Initialize and save given model
	model = BertModelModular(config_new)
	model.from_pretrained(args.pretrained_dir)
	model.save_pretrained(output_dir)

	# Finetune for all GLUE tasks
	glue_scores = {}
	score = 0

	for task in GLUE_TASKS:

		print(f'Finetuning on GLUE dataset: {task.upper()}')

		autotune = args.autotune # and not (task=='qqp' or task == 'qnli')
		training_args = get_training_args(output_dir, os.path.join(output_dir, task), task, autotune, args.autotune_trials)
		metrics = run_glue(training_args)
		print(metrics)

		if task == 'cola':

			glue_scores[task] = metrics['eval_matthews_correlation']
			task_score = glue_scores[task]

		elif task == 'stsb':

			glue_scores[task+'_spearman'] = metrics['eval_spearmanr']
			glue_scores[task+'_pearson'] = metrics['eval_pearson']
			task_score = max(metrics['eval_spearmanr'], metrics['eval_pearson']) # (metrics['eval_spearmanr']+metrics['eval_pearson'])/2.0

		elif task == 'mrpc' or task == 'qqp':

			glue_scores[task+'_accuracy'] = metrics['eval_accuracy']
			glue_scores[task+'_f1'] = metrics['eval_f1']
			task_score = max(metrics['eval_accuracy'], metrics['eval_f1']) # (metrics['eval_accuracy']+metrics['eval_f1'])/2.0

		elif task in ["sst2", "mnli",  "qnli", "rte", "wnli"]:

			glue_scores[task] = metrics['eval_accuracy']
			task_score = metrics['eval_accuracy']
				
		if task != 'wnli': score += task_score

	glue_scores['glue_score'] = score*1.0/8.0
						
	print(f"GLUE score for model: {score*1.0/8.0}")

	os.makedirs(output_dir, exist_ok=True)

	json.dump(glue_scores, open(os.path.join(output_dir, 'all_results.json'), 'w+'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--pretrained_dir',
		metavar='',
		type=str,
		help='path where the pretrained model is saved')
	parser.add_argument('--autotune',
		dest='autotune',
		action='store_true')
	parser.add_argument('--autotune_trials',
		metavar='',
		type=int,
		help='number of trials for optuna',
		default=20)
	parser.set_defaults(autotune=False)

	args = parser.parse_args()

	main(args)

