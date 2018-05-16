# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/switch_attractor_esn
# Description : NARMA 30 prediction with ESN.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Imports
import torch.utils.data
import dataset
from tools import settings
import echotorch.nn as etnn
from torch.autograd import Variable
import torchlanguage.transforms as transforms
import tools.functions
import tools.settings
import os
import echotorch
import socket


# Argument
args = tools.functions.argument_parser_execution()
print(args)
# Dataset info
data_infos = tools.functions.data_info(args.input_dataset)
print(u"1")
# Collection info
collection_info = dataset.TIRAAuthorIdentificationDataset.collection_infos(args.input_dataset)
print(u"2")
# Log collection info
print(collection_info)
print(u"3")
# Last lang
lang_lang = ""
transformer = None
w = None
print(u"4")
# For each problem
for problem_description in collection_info:
    # Problem name
    problem_name = problem_description['problem-name']
    problem_lang = problem_description['language']
    problem_encoding = problem_description['encoding']

    # Log
    print(u"Working on {} ({})".format(problem_name, problem_lang))

    # Transformer and W
    if lang_lang != problem_lang:
        if problem_lang == 'en' or problem_lang == 'fr':
            transformer = transforms.Compose([
                # transforms.RemoveLines(),
                transforms.GloveVector(model=tools.settings.lang_models[socket.gethostname()][problem_lang])
            ])
        else:
            transformer = transforms.Compose([
                # transforms.RemoveLines(),
                transforms.Token(model=tools.settings.lang_spacy_models[problem_lang],
                                 lang=tools.settings.lang_models_lang[problem_lang]),
                transforms.GensimModel(model_path=tools.settings.lang_models[socket.gethostname()][problem_lang])
            ])
        # end if
        w = torch.load(open('W/' + problem_lang + '.pth', 'rb'))
    # end if
    print(u"Author identification training dataset")
    # Author identification training dataset
    pan18loader_training = torch.utils.data.DataLoader(
        dataset.TIRAAuthorIdentificationDataset(
            root=args.input_dataset,
            transform=transformer,
            problem_name=problem_name,
            encoding=problem_encoding,
            train=True
        ),
        batch_size=1, shuffle=True
    )
    print(u"Author identification unknown dataset")
    # Author identification unknown dataset
    pan18loader_unknown = torch.utils.data.DataLoader(
        dataset.TIRAAuthorIdentificationDataset(
            root=args.input_dataset,
            transform=transformer,
            problem_name=problem_name,
            encoding=problem_encoding,
            train=False
        ),
        batch_size=1, shuffle=True
    )
    print(u"Authors")
    # Authors
    author_to_idx = dict()
    idx_to_author = dict()
    for idx, author in enumerate(pan18loader_training.dataset.authors):
        author_to_idx[author] = idx
        idx_to_author[idx] = author
    # end for

    # Number of authors
    n_authors = len(author_to_idx)
    print(n_authors)
    # ESN cell
    esn = etnn.BDESN(
        input_dim=settings.lang_models_dim[problem_lang],
        hidden_dim=settings.reservoir_size,
        output_dim=n_authors,
        spectral_radius=settings.spectral_radius,
        sparsity=settings.input_sparsity,
        input_scaling=settings.input_scaling,
        w_sparsity=settings.w_sparsity,
        learning_algo='inv',
        leaky_rate=settings.leaky_rate,
        w=w
    )

    # Get training data for this fold
    for i, data in enumerate(pan18loader_training):
        print(u"Learning from {}".format(pan18loader_training.dataset.last_text))
        # Inputs and labels
        inputs, labels = data

        # Create time labels
        author_id = author_to_idx[labels[0]]
        tag_vector = torch.zeros(1, inputs.size(1), n_authors)
        tag_vector[0, :, author_id] = 1.0

        # To variable
        inputs, time_labels = Variable(inputs), Variable(tag_vector)

        # Accumulate xTx and xTy
        esn(inputs, time_labels)
    # end for

    # Finalize training
    esn.finalize()

    # Results
    results = dict()

    # Get test data
    for i, data in enumerate(pan18loader_unknown):
        print(u"Evaluating {}".format(pan18loader_unknown.dataset.last_text))
        # Inputs and labels
        inputs, _ = data

        # To variable
        inputs = Variable(inputs)

        # Predict
        y_predicted = esn(inputs)

        # Normalized
        y_predicted -= torch.min(y_predicted)
        y_predicted /= torch.max(y_predicted) - torch.min(y_predicted)

        # Sum to one
        sums = torch.sum(y_predicted, dim=2)
        for t in range(y_predicted.size(1)):
            y_predicted[0, t, :] = y_predicted[0, t, :] / sums[0, t]
        # end for

        # Max average through time
        y_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=1)

        # Problem file
        problem_file = pan18loader_unknown.dataset.last_text

        # Save results
        results[problem_file] = idx_to_author[int(y_predicted[0])]
    # end for

    # Save result in file
    tools.functions.save_results(
        os.path.join(args.output_dir, "answers-{}.json".format(problem_name)),
        results
    )

    # Lang lang
    lang_lang = problem_lang

    # Remove ESN
    esn = None
# end for
