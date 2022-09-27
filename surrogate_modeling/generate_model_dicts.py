# Generate a dataset file with all possible model dictionaries and their hashes

# Author : Shikhar Tuli

import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')

from utils import graph_util
from utils import print_util as pu

import yaml
import json
import itertools
import argparse
from tqdm import tqdm


GENERATE_MODELS = False


def main():
    """Generate heterogeneous model dictionaries for the selected design space"""
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--design_space_file',
        metavar='', 
        type=str, 
        default='./design_space/design_space.yaml',
        help='path to yaml file for the design space')
    parser.add_argument('--dataset_file',
        metavar='',
        type=str,
        default='dataset.json',
        help='path to store the dataset')

    args = parser.parse_args()

    design_space = yaml.safe_load(open(args.design_space_file))

    dataset = {}

    feed_forward_ops = []
    for num_stacks in design_space['number_of_feed-forward_stacks']:
        feed_forward_ops.extend([list(tup) for tup in itertools.product(design_space['feed-forward_hidden'], repeat=num_stacks)])

    print(len(feed_forward_ops))

    attention_types = []
    for op_type in design_space['operation_types']:
        for op_param in design_space['operation_parameters'][op_type]:
            attention_types.append(op_type + '_' + str(op_param))

    print(len(attention_types))

    attention_ops = []
    for num_heads in design_space['num_heads']:
        attention_ops.extend([list(tup) for tup in itertools.combinations_with_replacement(attention_types, num_heads)])
        print(len([list(tup) for tup in itertools.combinations_with_replacement(attention_types, num_heads)]))

    print(len(attention_ops))

    count = 0

    for encoder_layers in tqdm(design_space['encoder_layers'], desc='Generating models'):
        model_dict = {'l': encoder_layers, 'o': [], 'h': [], 'f': []}

        count += len(feed_forward_ops) ** encoder_layers * len(design_space['hidden_size']) ** encoder_layers * len(attention_ops) ** encoder_layers

        if not GENERATE_MODELS: continue

        for feed_forward in itertools.combinations_with_replacement(feed_forward_ops, encoder_layers):
            for hidden_dim in itertools.combinations_with_replacement(design_space['hidden_size'], encoder_layers):
                for ops in itertools.combinations_with_replacement(attention_ops, encoder_layers):
                    # Add dimension of each attention operation based on hidden dimension of each layer
                    ops = list(ops); hidden_dim = list(hidden_dim); feed_forward = list(feed_forward)

                    for i, h in enumerate(hidden_dim):
                        ops[i] = [op + '_' + str(h) for op in ops[i]] 

                    model_dict['o'] = ops; model_dict['h'] = hidden_dim; model_dict['f'] = feed_forward

                    model_graph = graph_util.model_dict_to_graph(model_dict)
                    model_hash = graph_util.hash_graph(*model_graph, model_dict=model_dict)

                    dataset[model_hash] = model_dict

    print(f'{count: 0.5E} possible models in the dataset.')

    hashes = dataset.keys()

    assert len(hashes) == len(set(hashes)), 'Duplicate hashes found!'

    print(f'{pu.bcolors.OKGREEN}{len(hashes): 0.5E} models generated!{pu.bcolors.ENDC}')

    if GENERATE_MODELS: json.dump(dataset, open(args.dataset_file, 'w+'))


if __name__ == '__main__':
    main()




