# Generate a dataset file with all possible model dictionaries and their hashes

# Author : Shikhar Tuli

import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

from utils import graph_util
from utils import print_util as pu

import itertools
import numpy as np


def _get_possible_ops(design_space: dict):
    """Get possible operations
    
    Args:
        design_space (dict): design space dictionary

    Returns:
        feed_forward_ops (list), attention_ops (list): possible operations
    """

    # Get possible feed-forward operations
    feed_forward_ops = []
    for num_stacks in design_space['number_of_feed-forward_stacks']:
        feed_forward_ops.extend([list(tup) for tup in itertools.combinations_with_replacement(design_space['feed-forward_hidden'], num_stacks)])

    # Get possible attention types
    attention_types = []
    for op_type in design_space['operation_types']:
        for op_param in design_space['operation_parameters'][op_type]:
            attention_types.append(op_type + '_' + str(op_param))

    # Get possible attention operations
    attention_ops = []
    for num_heads in design_space['num_heads']:
        attention_ops.extend([list(tup) for tup in itertools.combinations_with_replacement(attention_types, num_heads)])

    return feed_forward_ops, attention_ops


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
            + 1 # for attention operations 
           )

    feed_forward_ops, attention_ops = _get_possible_ops(design_space)
    
    embedding = [0 for i in range(embedding_length)]

    embedding[0] = design_space['encoder_layers'].index(model_dict['l'])

    for layer in range(model_dict['l']):
        hidden_dim_idx = design_space['hidden_size'].index(model_dict['h'][layer])
        embedding[layer * 3 + 1] = hidden_dim_idx

        feed_forward_idx = feed_forward_ops.index(model_dict['f'][layer])
        embedding[layer * 3 + 2] = feed_forward_idx

        attn_ops = []
        for i, op in enumerate(model_dict['o'][layer]):
            op_type, op_param, _ = op.split('_')
            attn_ops.append(op_type + '_' + op_param)
        embedding[layer * 3 + 3] = attention_ops.index(attn_ops)

    return embedding


def embedding_to_model_dict(embedding: list, design_space: dict):
    """Convert an embedding to model dictionary
    
    Args:
        membedding (list): embedding for the given model dictionary
        design_space (dict): design space dictionary

    Returns:
        model_dict (dict): model dictionary
    """

    # First we find the embedding length based on the design space
    embedding_length = 1 + max(design_space['encoder_layers']) \
        * (1 # for hidden dimension
            + 1 # for feed-forward stack 
            + 1 # for attention operations 
           )

    feed_forward_ops, attention_ops = _get_possible_ops(design_space)

    model_dict = {'l': design_space['encoder_layers'][embedding[0]], 
        'o': [[] for i in range(design_space['encoder_layers'][embedding[0]])], 
        'h': [], 'f': []}

    for layer in range(model_dict['l']):
        model_dict['h'].append(design_space['hidden_size'][embedding[layer * 3 + 1]])

        model_dict['f'].append(feed_forward_ops[embedding[layer * 3 + 2]])

        attn_ops = attention_ops[embedding[layer * 3 + 3]]
        model_dict['o'][layer] = [attn + '_' + f'{model_dict["h"][-1]//len(attn_ops)}' for attn in attn_ops] 

    return model_dict


def get_embedding_bounds(design_space: dict, type: str = 'all'):
    """Get bounds for Sobol sampling
    
    Args:
        design_space (dict): design space dictionary
        type (str, optional): bounds for model types required. In {'all', 'narrow', 'wide'}
    
    Returns:
        bounds (list): list of tuples with lower and upper bounds
    """

    # First we find the embedding length based on the design space
    embedding_length = 1 + max(design_space['encoder_layers']) \
        * (1 # for hidden dimension
            + 1 # for feed-forward stack 
            + 1 # for attention operations 
           )

    feed_forward_ops, attention_ops = _get_possible_ops(design_space)

    # Get index for median number of attention heads
    median_num_heads_idx = 0
    median_num_heads = design_space['num_heads'][len(design_space['num_heads'])//2]
    for i, attn_ops in enumerate(attention_ops):
        if len(attn_ops) == median_num_heads: 
            median_num_heads_idx = i - 1
            break

    bounds = [() for i in range(embedding_length)]

    bounds[0] = (0, len(design_space['encoder_layers']) - 1)

    for layer in range(max(design_space['encoder_layers'])):
        bounds[layer * 3 + 1] = (0, len(design_space['hidden_size']) - 1)
        bounds[layer * 3 + 2] = (0, len(feed_forward_ops) - 1)

        if type == 'all':
            bounds[layer * 3 + 3] = (0, len(attention_ops) - 1)
        elif type == 'narrow':
            bounds[layer * 3 + 3] = (0, median_num_heads_idx)
        else:
            bounds[layer * 3 + 3] = (median_num_heads_idx + 1, len(attention_ops) - 1)

    return bounds


def is_valid_embedding(embedding: list, design_space: dict):
    """Test if an embedding is valid or not
    
    Args:
        embedding (list): embedding for the given model dictionary
        design_space (dict): design space dictionary

    Returns:
        valid (bool): whether the embedding is valid or not
    """

    # All entries beyond embedding[0] layers should be zero
    if np.count_nonzero(embedding[design_space['encoder_layers'][embedding[0]] * 3 + 1:]) > 0:
        return False

    # Test if an embedding can form a valid model dictionary, a model graph and is hashable
    try:
        model_dict = embedding_to_model_dict(embedding, design_space)
        model_graph = graph_util.model_dict_to_graph(model_dict)
        model_hash = graph_util.hash_graph(*model_graph, model_dict=model_dict)
        return True
    except:
        return False


def get_nearest_valid_embedding(embedding: list, design_space: dict):
    """Get the nearest valid embeddding for the given embedding
    
    Args:
        embedding (list): embedding for the given model dictionary
        design_space (dict): design space dictionary

    Returns:
        valid_embedding (list): valid embedding from the given embedding
    """

    embedding[design_space['encoder_layers'][embedding[0]] * 3 + 1:] = \
        [0 for i in range(len(embedding[design_space['encoder_layers'][embedding[0]] * 3 + 1:]))]

    assert is_valid_embedding(embedding, design_space)

    return embedding


