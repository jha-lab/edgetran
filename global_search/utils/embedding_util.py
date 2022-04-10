# Generate a dataset file with all possible model dictionaries and their hashes

# Author : Shikhar Tuli

import os
import sys

import itertools
import numpy as np


def model_dict_to_embedding(model_dict: dict, design_space: dict):
    """Convert a model dictionary to corresponding embedding
    
    Args:
        model_dict (dict): model dictionary (based on new heterogeneous format)
        design_space (dict): design space dictionary
    
    Returns:
        embedding (list): embedding for the given model dictionary
    """

    # First we find the embedding length based on the design space
    embedding_length = 1 + max(design_space['encoder_layers']) \
        * (1 # for hidden dimension
            + 1 # for feed-forward stack 
            + max(design_space['num_heads']) # for each attention head 
           )

    # Get possible feed-forward operations
    feed_forward_ops = []
    for num_stacks in design_space['number_of_feed-forward_stacks']:
        feed_forward_ops.extend([list(tup) for tup in itertools.combinations_with_replacement(design_space['feed-forward_hidden'], num_stacks)])

    # Get possible attention operations
    attention_types = []
    for op_type in design_space['operation_types']:
        for op_param in design_space['operation_parameters'][op_type]:
            attention_types.append(op_type + '_' + str(op_param))
    
    embedding = [0 for i in range(embedding_length)]

    embedding[0] = design_space['encoder_layers'].index(model_dict['l'])

    for layer in range(model_dict['l']):
        hidden_dim_idx = design_space['hidden_size'].index(model_dict['h'][layer])
        embedding[layer * (max(design_space['num_heads']) + 2) + 1] = hidden_dim_idx

        feed_forward_idx = feed_forward_ops.index(model_dict['f'][layer])
        embedding[layer * (max(design_space['num_heads']) + 2) + 2] = feed_forward_idx

        for i, op in enumerate(model_dict['o'][layer]):
            op_type, op_param, _ = op.split('_')
            embedding[layer * (max(design_space['num_heads']) + 2) + 3 + i] = attention_types.index(op_type + '_' + op_param) + 1

    return embedding





