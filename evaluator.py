#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import argparse
from tools import evaluate_all


# Parser
parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2018')
parser.add_argument('-i', type=str, help='Path to evaluation collection')
parser.add_argument('-a', type=str, help='Path to answers folder')
parser.add_argument('-o', type=str, help='Path to output files')
args = parser.parse_args()

# Check collection path
if not args.i:
    print('ERROR: The collection path is required')
    parser.exit(1)
# end if

# Check answers path
if not args.a:
    print('ERROR: The answers folder is required')
    parser.exit(1)
# end if

# Check output path
if not args.o:
    print('ERROR: The output path is required')
    parser.exit(1)
# end if

# Evaluate
evaluate_all(args.i, args.a, args.o)
