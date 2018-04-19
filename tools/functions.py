# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchvision import transforms
import argparse
import dataset
import torch
import settings
import os
import codecs
import json

#################
# Arguments
#################


# Argument parser for training model
def argument_parser_training_model():
    """
    Tweet argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Authorship Identification challenge")

    # Argument
    parser.add_argument("--lang", type=str, help="Problem language (en, fr, it, pl, sp)", default='en')
    args = parser.parse_args()

    # Use CUDA?
    return args
# end argument_parser_training_model


# Execution argument parser
def argument_parser_execution():
    """
    Execution argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Author Profiling main program")

    # Argument
    parser.add_argument("--input-dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--output-dir", type=str, help="Where to put results", required=True)
    parser.add_argument("--input-run", type=str, help="Input run", required=True)
    # parser.add_argument("--w", type=str, help="Reservoir matrix", required=True)
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
# end argument_parser_execution


#################
# Results
#################


# Save results
def save_results(output_file, results_dict):
    """
    Save results
    :param results_dict:
    :return:
    """
    # JSON list
    json_results = list()

    # For each result
    for problem_file in results_dict.keys():
        predicted_author = "candidate" + str(results_dict[problem_file]).zfill(5)
        json_results.append({"unknown-text": problem_file, "predicted-author": predicted_author})
    # end for

    # Write
    json.dump(json_results, open(output_file, 'w'))
# end save_results
