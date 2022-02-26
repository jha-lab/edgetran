# Transformer dataset class

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

from treelib import Tree, Node
import json

from utils import print_util as pu


MODES = ['grow_attn_head', 'grow_ffnn', 'prune_attn_head', 'prune_ffnn', 'prune_encoder_layer']


class TxfNode(object):
	def __init__(self, model_hash: str, mode: str, loss=None):
		"""Node corresponding to every transformer in the dataset
		
		Args:
			model_hash (str): hash of the given model
			mode (str): mode of change from parent, None if root
			loss (float, optional): lowest value in losses
		"""
		if mode is not None:
			assert mode in MODES, f'Mode should in {MODES}'

		# Set node parameters
		self.model_hash = model_hash
		self.mode = mode
		self.loss = loss

	def __repr__(self):
		return str(self.__dict__)


class TxfDataset(object):
	def __init__(self, dataset_file: str, models_dir: str, debug=False):
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

		if os.path.exists(self.dataset_file):
			self.load_dataset(self.dataset_file)
			if self.debug:
				print(f'{pu.bcolors.OKGREEN}Loaded dataset from file{pu.bcolros.ENDC}')

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
		self.dataset = self._load_tree(Tree(), dataset_dict)

	def save_dataset(self):
		"""Save the dataset to file"""
		json.dump(eval(str(self.dataset.to_dict(with_data=True))), open(self.dataset_file, 'w+'))

	def to_dict(self, with_data=True):
		"""Get dictionary object of tree dataset
		
		Args:
		    with_data (bool): with_data attribute for treelib.Tree
		
		Returns:
		    dict: dictionary object of current tree dataset
		"""
		return eval(str(self.dataset.to_dict(with_data=with_data)))

