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
import echotorch.nn as etnn
from torch.autograd import Variable
import echotorch.utils
import torchlanguage.transforms as transforms
import torchlanguage.models
import matplotlib.pyplot as plt
import tools.functions
import tools.settings
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

# Experiment settings
reservoir_size = 300
spectral_radius = 0.95
input_sparsity = 0.1
w_sparsity = 0.1
input_scaling = 0.5
n_test = 5
n_samples = 1

# Argument
args = tools.functions.argument_parser_training_model()

# Load feature selector
feature_selector = torchlanguage.models.CCSAA(text_length=20, vocab_size=84, embedding_dim=50, n_classes=15)
feature_selector.load_state_dict(torch.load(open(args.feature_selector, 'rb')))
feature_selector.linear = etnn.Identity()
if args.cuda:
    feature_selector.cuda()
# end if
feature_selector_voc = torch.load(open(args.feature_selector_voc, 'rb'))

# Transforms
transform = transforms.Compose([
    transforms.Character(),
    transforms.ToIndex(token_to_ix=feature_selector_voc),
    transforms.MaxIndex(max_id=83),
    transforms.ToNGram(n=20, overlapse=True),
    transforms.Reshape((-1, 20)),
    transforms.ToCUDA(),
    transforms.FeatureSelector(model=feature_selector, n_features=150, to_variable=True),
    transforms.ToCPU(),
    transforms.Normalize(mean=-5.08, std=0.3294)
])

# Results
parameter_averages = np.zeros(n_test)
parameter_max = np.zeros(n_test)

# For each leaky rate values
index = 0
for leaky_rate in np.linspace(0.6, 1.0, n_test):
    # Log
    print(u"Leaky rate : {}".format(leaky_rate))

    # Samples average
    samples_average = np.array([])

    # For each samples
    for n in range(n_samples):
        # Create W matrix
        w = etnn.ESNCell.generate_w(reservoir_size, w_sparsity)

        # Sample average
        single_sample_average = np.array([])

        # For each problem
        for problem in np.arange(1, 3):
            # Truth and prediction
            y_true = np.array([])
            y_pred = np.array([])

            # Author identification training dataset
            pan18loader_training = torch.utils.data.DataLoader(
                dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=problem, lang=args.lang),
                batch_size=1, shuffle=True
            )

            # Author identification test dataset
            pan18loader_test = torch.utils.data.DataLoader(
                dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=problem, train=False, lang=args.lang),
                batch_size=1, shuffle=True
            )

            # Authors
            author_to_idx = dict()
            for idx, author in enumerate(pan18loader_training.dataset.authors):
                author_to_idx[author] = idx
            # end for

            # Number of authors
            n_authors = len(author_to_idx)

            # ESN cell
            esn = etnn.BDESN(
                input_dim=150,
                hidden_dim=reservoir_size,
                output_dim=n_authors,
                spectral_radius=spectral_radius,
                sparsity=input_sparsity,
                input_scaling=input_scaling,
                learning_algo='inv',
                leaky_rate=leaky_rate,
                w=w
            )

            # Get training data for this fold
            for i, data in enumerate(pan18loader_training):
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

            # Get test data
            for i, data in enumerate(pan18loader_test):
                # Inputs and labels
                inputs, labels = data

                # Author id
                author_id = author_to_idx[labels[0]]

                # To variable
                inputs, label = Variable(inputs), Variable(torch.LongTensor([author_id]))

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

                # Add to array
                y_true = np.append(y_true, int(label[0]))
                y_pred = np.append(y_pred, int(y_predicted[0]))
            # end for

            # F1
            sample_f1_score = f1_score(y_true, y_pred, average='macro')

            # Save result
            single_sample_average = np.append(single_sample_average, [sample_f1_score])

            # Reset ESN
            esn.reset()
        # end for
        print(u"\t\tSample f1 score : {}".format(np.average(single_sample_average)))
        # Save results
        samples_average = np.append(samples_average, [np.average(single_sample_average)])
    # end for

    # Show result
    print(u"\tMacro average F1 score : {} (max {})".format(np.average(samples_average), np.max(samples_average)))

    # Save results
    parameter_averages[index] = np.average(samples_average)
    parameter_max[index] = np.max(samples_average)
    index += 1
# end for

# Show
print(parameter_averages)
print(parameter_max)
print(np.max(parameter_max))

# Show results
plt.plot(parameter_averages)
plt.show()
