# Utility functions for embeddings and model sampling

# Author : Shikhar Tuli

import os
import sys

sys.path.append('../../txf_design-space/')
sys.path.append('../../txf_design-space/flexibert')

from embeddings.utils import graph_util, print_util as pu

import itertools
import numpy as np
import collections

from skopt.sampler import Sobol, Lhs, Halton, Hammersly
from skopt.space import Space
import hashlib


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
        feed_forward_ops.extend([list(tup) for tup in itertools.product(design_space['feed-forward_hidden'], repeat=num_stacks)])

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
        model_dict (dict): model dictionary (based on new heterogeneous format in FlexiBERT 2.0)
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
        embedding (list): embedding for the given transformer
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


def embedding_to_config(embedding: list, design_space: dict):
    """Convert an embedding to accelerator config
    
    Args:
        embedding (list): embedding for the given accelerator
        design_space (dict): design space dictionary

    Returns:
        config (dict): configuration of the accelerator
    """

    config = {'tile': {'tile_b': None, 'tile_x': None , 'tile_y': None}, 'non_linearity': 'gelu', 'pe': None, 'lanes_per_pe': None, 'mac_per_lane': None, 'softmax_per_pe': None, 'batch_size': None, 'activation_buffer_size': None, 'weight_buffer_size': None, 'mask_buffer_size': None, 'loop_unrolling': None, 'main_memory': {'type': None, 'mode': None, 'banks': None, 'ranks': None, 'channels': None}, 'scheduler': {'compute_ops': {'tiled': True, 'batch_size': None}, 'memory_ops': {'tiled': False, 'batch_size': 1}}}

    i = 0
    for tile in config['tile'].keys():
        config['tile'][tile] = embedding[i]
        i += 1

    for decision in ['pe', 'lanes_per_pe', 'mac_per_lane', 'softmax_per_pe', 'batch_size', 'activation_buffer_size', 'weight_buffer_size', 'mask_buffer_size']:
        config[decision] = embedding[i]
        i += 1

    ordered_configs = [('rram', 16, 2, 2), ('rram', 8, 2, 4), ('rram', 4, 2, 8), ('rram', 2, 2, 16), ('rram', 32, 2, 1), ('rram', 1, 2, 32), ('dram', 16, 2, 2), ('dram', 8, 2, 4), ('dram', 32, 2, 1), ('dram', 16, 4, 1), ('hbm', 32, 1, 4)] 

    config['main_memory']['type'] = ordered_configs[embedding[-1]][0]
    config['main_memory']['mode'] = 'lb' if config['main_memory']['type'] == 'dram' else 'hb'
    config['main_memory']['banks'], config['main_memory']['ranks'], config['main_memory']['channels'] = ordered_configs[embedding[-1]][1], ordered_configs[embedding[-1]][2], ordered_configs[embedding[-1]][3]

    # Determine batch size based on heuristic
    config['scheduler']['compute_ops']['batch_size'] = config['pe'] // 4

    config['loop_unrolling'] = 'b_i_j_k'

    return config


def config_to_embedding(config: dict, design_space: dict):
    """Convert an accelerator config to corresponding embedding
    
    Args:
        config (dict): configuration of the accelerator
        design_space (dict): design space dictionary

    Returns:
        embedding (list): embedding for the given accelerator
    """

    embedding = [0 for _ in range(12)]

    i = 0
    for tile in config['tile'].keys():
        embedding[i] = config['tile'][tile]
        i += 1

    for decision in ['pe', 'lanes_per_pe', 'mac_per_lane', 'softmax_per_pe', 'batch_size', 'activation_buffer_size', 'weight_buffer_size', 'mask_buffer_size']:
        embedding[i] = config[decision]
        i += 1

    main_memory_config = (config['main_memory']['type'], config['main_memory']['banks'], config['main_memory']['ranks'], config['main_memory']['channels'])

    ordered_configs = [('rram', 16, 2, 2), ('rram', 8, 2, 4), ('rram', 4, 2, 8), ('rram', 2, 2, 16), ('rram', 32, 2, 1), ('rram', 1, 2, 32), ('dram', 16, 2, 2), ('dram', 8, 2, 4), ('dram', 32, 2, 1), ('dram', 16, 4, 1), ('hbm', 32, 1, 4)]

    for i, c in enumerate(ordered_configs):
        if all([(a == b) for a, b in zip(main_memory_config, c)]): 
            embedding[-1] = i
            break

    assert is_valid_embedding(embedding, design_space, 'accelerator')

    return embedding


def get_embedding_bounds(design_space: dict, model_types: str = 'all', space: str = 'transformer'):
    """Get bounds for Sobol sampling
    
    Args:
        design_space (dict): design space dictionary
        model_types (str, optional): bounds for model types required. In {'all', 'narrow', 'wide'}
        space (str, optional): space in consideration. In {'transformer', 'accelerator'}
    
    Returns:
        bounds (list): list of tuples with lower and upper bounds
    """

    assert space in ['transformer', 'accelerator']

    if space == 'transformer':
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

            if model_types == 'all':
                bounds[layer * 3 + 3] = (0, len(attention_ops) - 1)
            elif model_types == 'narrow':
                bounds[layer * 3 + 3] = (0, median_num_heads_idx)
            else:
                bounds[layer * 3 + 3] = (median_num_heads_idx + 1, len(attention_ops) - 1)
    else:
        embedding_length = 12

        bounds = [() for i in range(embedding_length)]

        i = 0
        for tile in design_space['tile'].keys():
            bounds[i] = (min(design_space['tile'][tile]), max(design_space['tile'][tile]))
            i += 1

        for decision in ['pe', 'lanes_per_pe', 'mac_per_lane', 'softmax_per_pe', 'batch_size', 'activation_buffer_size', 'weight_buffer_size', 'mask_buffer_size']:
            bounds[i] = (min(design_space[decision]), max(design_space[decision]))
            i += 1
        
        bounds[-1] = (0, 10)

    return bounds


def is_valid_embedding(embedding: list, design_space: dict, space: str = 'transformer'):
    """Test if an embedding is valid or not
    
    Args:
        embedding (list): embedding for the given model dictionary
        design_space (dict): design space dictionary
        space (str, optional): space in consideration. In {'transformer', 'accelerator'}

    Returns:
        valid (bool): whether the embedding is valid or not
    """

    assert space in ['transformer', 'accelerator']

    if space == 'transformer':
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
    else:
        embedding_length = 12

        i = 0
        for tile in design_space['tile'].keys():
            if embedding[i] not in design_space['tile'][tile]: return False
            i += 1

        for decision in ['pe', 'lanes_per_pe', 'mac_per_lane', 'softmax_per_pe', 'batch_size', 'activation_buffer_size', 'weight_buffer_size', 'mask_buffer_size']:
            if embedding[i] not in design_space[decision]: return False
            i += 1

        if embedding[-1] not in np.arange(0, 11): return False

        return True


def get_nearest_valid_embedding(embedding: list, design_space: dict, space: str = 'transformer'):
    """Get the nearest valid embeddding for the given embedding
    
    Args:
        embedding (list): embedding for the given model dictionary
        design_space (dict): design space dictionary
        space (str, optional): space in consideration. In {'transformer', 'accelerator'}

    Returns:
        valid_embedding (list): valid embedding from the given embedding
    """

    assert space in ['transformer', 'accelerator']

    if space == 'transformer':
        embedding[design_space['encoder_layers'][embedding[0]] * 3 + 1:] = \
            [0 for i in range(len(embedding[design_space['encoder_layers'][embedding[0]] * 3 + 1:]))]
    else:
        embedding_length = 12

        i = 0
        for tile in design_space['tile'].keys():
            embedding[i] = min(design_space['tile'][tile], key=lambda x : abs(x - embedding[i]))
            i += 1

        for decision in ['pe', 'lanes_per_pe', 'mac_per_lane', 'softmax_per_pe', 'batch_size', 'activation_buffer_size', 'weight_buffer_size', 'mask_buffer_size']:
            embedding[i] = min(design_space[decision], key=lambda x : abs(x - embedding[i]))
            i += 1

        embedding[-1] = min(np.arange(0, 11), key=lambda x : abs(x - embedding[-1]))

    assert is_valid_embedding(embedding, design_space, space)

    return embedding


def get_model_type(model_dict: dict, design_space: dict):
    """Get the type of model among four broac categories
    
    Args:
        model_dict (dict): model dictionary (based on new heterogeneous format)
        design_space (dict): design space dictionary

    Returns:
        model_type (str): 
    """

    median_encoder_layers = design_space['encoder_layers'][len(design_space['encoder_layers'])//2]
    median_num_heads = design_space['num_heads'][len(design_space['num_heads'])//2]

    depth = model_dict['l']
    max_num_heads = max([len(attn_ops) for attn_ops in model_dict['o']])

    model_type = 'shallow_' if depth < median_encoder_layers else 'deep_'
    model_type += 'narrow' if max_num_heads < median_num_heads else 'wide'

    return model_type


def get_samples(design_space: dict, num_samples: int, sampling_method='Lhs', space: str = 'transformer', debug=False):
    """Get the embeddings sampled using the given low-discrepancy sampling method
    
    Args:
        design_space (dict): design space dictionary
        num_samples (int): number of samples 
        sampling_method (str, optional): low-discrepancy sampling method in ['Sobol', 'Lhs', 'Halton', Hammersly', 'Random']
        space (str, optional): space in consideration. In {'transformer', 'accelerator'}
        debug (bool, optional): to print debugging output
    
    Returns
        samples_dict (dict): dictionary of sampled models
    """

    if space == 'accelerator':
        assert sampling_method == 'Random', 'Only random sampling is supported with the accelerator design space'

    if debug: print(f'Generating {num_samples} samples using the {sampling_method} sampler...')

    if sampling_method != 'Random':
        narrow_embedding_bounds = get_embedding_bounds(design_space, model_types='narrow')
        wide_embedding_bounds = get_embedding_bounds(design_space, model_types='wide')
    else:
        embedding_bounds = get_embedding_bounds(design_space, model_types='all', space=space)
    
    if space == 'transformer':
        if sampling_method == 'Lhs':
            narrow_sampler = eval(f'{sampling_method}(criterion="ratio")')
            wide_sampler = eval(f'{sampling_method}(criterion="ratio")')
        elif sampling_method != 'Random':
            narrow_sampler = eval(f'{sampling_method}()')
            wide_sampler = eval(f'{sampling_method}()')
        else:
            sampler = Space(embedding_bounds)

        if sampling_method != 'Random':
            narrow_sampled_embeddings = eval(f'narrow_sampler.generate(narrow_embedding_bounds, {num_samples//2}, random_state=0)')
            wide_sampled_embeddings = eval(f'wide_sampler.generate(wide_embedding_bounds, {num_samples//2}, random_state=0)')

            narrow_valid_embeddings = [get_nearest_valid_embedding(embedding, design_space) for embedding in narrow_sampled_embeddings]
            wide_valid_embeddings = [get_nearest_valid_embedding(embedding, design_space) for embedding in wide_sampled_embeddings]

            narrow_model_dicts = [embedding_to_model_dict(embedding, design_space) for embedding in narrow_valid_embeddings]
            wide_model_dicts = [embedding_to_model_dict(embedding, design_space) for embedding in wide_valid_embeddings]

            narrow_model_types = [get_model_type(model_dict, design_space) for model_dict in narrow_model_dicts]
            wide_model_types = [get_model_type(model_dict, design_space) for model_dict in wide_model_dicts]

            if debug: print(f'Narrow model types: {collections.Counter(narrow_model_types)}')
            if debug: print(f'Wide model types: {collections.Counter(wide_model_types)}')

            all_embeddings = narrow_valid_embeddings + wide_valid_embeddings
            all_model_dicts = narrow_model_dicts + wide_model_dicts
            all_model_types = narrow_model_types + wide_model_types
        else:
            sampled_embeddings = sampler.rvs(num_samples)

            valid_embeddings = [get_nearest_valid_embedding(embedding, design_space) for embedding in sampled_embeddings]
            model_dicts = [embedding_to_model_dict(embedding, design_space) for embedding in valid_embeddings]
            model_types = [get_model_type(model_dict, design_space) for model_dict in model_dicts]

            if debug: print(f'Model types: {collections.Counter(model_types)}')

            all_embeddings = valid_embeddings
            all_model_dicts = model_dicts
            all_model_types = model_types
        
        all_hashes = []
        samples_dict = {}

        assert len(all_embeddings) == len(all_model_dicts)

        for i in range(len(all_model_dicts)):
            model_dict = all_model_dicts[i]
            embedding = all_embeddings[i]
            model_type = all_model_types[i]

            model_graph = graph_util.model_dict_to_graph(model_dict)
            model_hash = graph_util.hash_graph(*model_graph, model_dict=model_dict)
            all_hashes.append(model_hash)

            samples_dict[model_hash] = {'model_dict': model_dict, 'model_type': model_type, 'embedding': embedding}

        assert len(set(all_hashes)) == len(all_model_dicts) 

    else:
        sampler = Space(embedding_bounds)
        sampled_embeddings = sampler.rvs(num_samples)

        valid_embeddings = [get_nearest_valid_embedding(embedding, design_space, space) for embedding in sampled_embeddings]

        samples_dict = {}

        for embedding in valid_embeddings:
            accelerator_hash = hashlib.md5(str(embedding).encode('utf-8')).hexdigest()
            samples_dict[accelerator_hash] = embedding

    return samples_dict
        

