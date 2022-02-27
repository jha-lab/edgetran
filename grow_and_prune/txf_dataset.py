# Transformer dataset class

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

import json
import numpy as np
from treelib import Tree, Node

from utils import print_util as pu


MODES = ['grow_attn_head', 'grow_ffnn', 'prune_attn_head', 'prune_ffnn', 'prune_encoder_layer']


class TxfNode(object):
	def __init__(self, model_hash: str, mode: str, loss=None, steps=None):
		"""Node corresponding to every transformer in the dataset
		
		Args:
			model_hash (str): hash of the given model
			mode (str): mode of change from parent, None if root
			loss (float, optional): lowest value in losses
			steps (int, optional): number of steps the model is trained
		"""
		if mode is not None:
			assert mode in MODES, f'Mode should be in {MODES}'

		# Set node parameters
		self.model_hash = model_hash
		self.mode = mode
		self.loss = loss
		self.steps = steps

	def __repr__(self):
		return str(self.__dict__)


class TxfDataset(object):
	def __init__(self, dataset_file=None, models_dir=None, debug=False):
		"""Wrapper class around treelib.Tree to load and store the transformer dataset
		
		Args:
			dataset_file (str): path to the dataset file (.json)
			models_dir (str): path to the models directory
			debug (bool, optional): debugging mode
		"""
		self.dataset_file = dataset_file
		self.models_dir = models_dir
		self.dataset = Tree()
		self.debug = debug
		self.next_best_idx = 0

		if os.path.exists(self.dataset_file):
			self.load_dataset(self.dataset_file)
			if self.debug:
				print(f'{pu.bcolors.OKGREEN}Loaded dataset from file{pu.bcolors.ENDC}')

	def _load_tree(self, tree: Tree, tree_dict: dict, parent=None):
		"""Recursive function to load the tree
		
		Args:
		    tree (Tree): treelib.Tree object
		    tree_dict (dict): tree dictionary loaded from dataset_file
		    parent (Node, optional): parent node to start with
		
		Returns:
		    tree: (Tree) a treelib.Tree object
		"""
		model_hash, model_value = list(tree_dict.items())[0]

		if parent is None:
			tree.create_node(tag=model_hash, identifier=model_hash, data=TxfNode(**model_value['data']))
			parent = tree.get_node(model_hash)
		else:
			tree.create_node(tag=model_hash, identifier=model_hash, parent=parent, data=TxfNode(**model_value['data']))

		for child in tree_dict[model_hash].get('children', []):  
			self._load_tree(tree, child, parent)

		return tree

	def load_dataset(self, dataset_file: str):
		"""Load the tree dataset
		
		Args:
			dataset_file (str): path to the dataset file (.json)
		"""
		dataset_dict = json.load(open(dataset_file, 'r'))
		self.dataset = self._load_tree(Tree(), dataset_dict['dataset'])
		self.dataset_file = dataset_dict['dataset_file']
		self.models_dir = dataset_dict['models_dir']
		self.next_best_idx = dataset_dict['next_best_idx']

	def save_dataset(self):
		"""Save the dataset to file"""
		json.dump({'dataset': self.to_dict(with_data=True), 
			'dataset_file': self.dataset_file,
			'models_dir': self.models_dir,
			'next_best_idx': self.next_best_idx}, open(self.dataset_file, 'w+'))

	def to_dict(self, with_data=True):
		"""Get dictionary object of tree dataset
		
		Args:
		    with_data (bool): with_data attribute for treelib.Tree
		
		Returns:
		    dict: dictionary object of current tree dataset
		"""
		return eval(str(self.dataset.to_dict(with_data=with_data)))

	def add_node(self, model_hash: str, mode: str, loss=None, steps=None, parent_model_hash=None):
		"""Add a TxfNode object to the current graph
		
		Args:
		    model_hash (str): hash of the given model
			mode (str): mode of change from parent, None if root
		    loss (float, optional): lowest value in losses
		    steps (int, optional): number of steps the model is trained
		    parent_model_hash (str, optional): hash of the parent model
		"""
		self.dataset.create_node(tag=model_hash, 
			identifier=model_hash, 
			parent=self.dataset.get_node(parent_model_hash) if parent_model_hash is not None else None, 
			data=TxfNode(model_hash, mode, loss, steps))

	def update_dataset(self, save_dataset=True):
		"""Update the dataset based on trained models in models_dir
		
		Args:
		    save_dataset (bool, optional): save dataset after update
		
		Returns:
		    float, str: best loss and hash
		"""
		best_loss, best_hash = np.inf, ''
		for model_hash in os.listdir(self.models_dir):
			if not os.path.exists(os.path.join(self.models_dir, model_hash, 'log_history.json')): 
				continue

			log_history = json.load(open(os.path.join(self.models_dir, model_hash, 'log_history.json'), 'r'))
			losses = [state['loss'] for state in log_history[:-1]]
			steps = [state['step'] for state in log_history[:-1]]
			
			self.dataset[model_hash].data.loss = min(losses)
			self.dataset[model_hash].data.steps = steps[-1]

			if self.dataset[model_hash].data.loss < best_loss:
				best_loss = self.dataset[model_hash].data.loss
				best_hash = model_hash

		if save_dataset: self.save_dataset()

		if self.debug:
			print(f'{pu.bcolors.OKBLUE}Model with best loss ({best_loss}) has hash: {best_hash}{pu.bcolors.ENDC}')
		
		return best_loss, best_hash

	def get_next_best_model(self):
		"""Get next best loss and hash

		Returns:
		    float, str: next best loss and hash
		"""
		self.next_best_idx += 1

		hash_losses = []
		for model_hash in os.listdir(models_dir):
			if not os.path.exists(os.path.join(self.models_dir, model_hash, 'log_history.json')): 
				continue

			log_history = json.load(open(os.path.join(self.models_dir, model_hash, 'log_history.json'), 'r'))
			losses = [state['loss'] for state in log_history[:-1]]
			
			hash_losses.append({'model_hash': model_hash, 'loss': min(losses)})

		hash_losses.sort(key=lambda x:x['loss'])

		return hash_losses[self.next_best_idx]['loss'], hash_losses[self.next_best_idx]['model_hash']

	def get_log_history(self, model_hash: str):
		"""Get log history of the given model

		Args:
		    model_hash (str): hash of the given model

		Returns:
			dict: log history of the given model
		"""
		return json.load(open(os.path.join(self.models_dir, model_hash, 'log_history.json'), 'r'))

	def get_model_dict(self, model_hash: str):
		"""Get model dictionary of the given model

		Args:
		    model_hash (str): hash of the given model

		Returns:
			dict: model dictionary of the model
		"""
		return json.load(open(os.path.join(self.models_dir, model_hash, 'model_dict.json'), 'r'))

	def get_parent_hash(self, model_hash: str):
		"""Get hash of the given model"""
		return self.dataset.parent(model_hash).data.model_hash

	def get_mode(self, model_hash: str):
		"""Get mode  of the given model"""
		return self.dataset.get_node(model_hash).data.mode

	def is_root(self, model_hash: str):
		"""Check if model is root"""
		return model_hash == self.dataset.root

	def has_children(self, model_hash: str):
		"""Check if model has children"""
		return len(self.dataset.children(model_hash)) > 0

	def show_dataset(self, data_property='loss'):
		"""Show the current dataset in tree format
		
		Args:
		    data_property (str, optional): data_property attribute of treelib.Tree
		"""
		self.dataset.show(data_property=data_property, idhidden=False, line_type='ascii-exr')

