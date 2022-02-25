# Transformer dataset class

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

from treelib import Tree
import json

from utils import print_util as pu


MODES = ['grow_attn_head', 'grow_ffnn', 'prune_attn_head', 'prune_ffnn', 'prune_encoder_layer']


class TxfNode(object):
	def __init__(self, model_hash: str, model_dict: dict, losses: list, mode: str):
		"""Node corresponding to every transformer in the dataset
		
		Args:
		    model_hash (str): hash of the given model
		    model_dict (dict): model dictionary of the given model
		    loss (float): losses for the given model 
		    mode (str): mode of change from parent
		"""
		assert mode in MODES, f'Mode should in {MODES}'

		# Set node parameters
		self.model_hash = model_hash
		self.model_dict = model_dict
		self.losses = losses
		self.mode = mode

		# Set lowest loss
		self.lowest_loss = min(losses)


		
		


