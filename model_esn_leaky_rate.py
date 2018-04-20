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
import matplotlib.pyplot as plt
import tools.functions
import tools.settings
import numpy as np
from sklearn.metrics import recall_score

# Experiment settings
reservoir_size = 1000
spectral_radius = 0.95
input_sparsity = 0.1
w_sparsity = 0.1
input_scaling = 0.5
n_test = 2
n_samples = 2

# Argument
args = tools.functions.argument_parser_training_model()

# Transformer
transformer = transforms.Compose([
    transforms.RemoveLines(),
    transforms.GloveVector(model=tools.settings.lang_models[args.lang])
])

# Results
parameter_averages = np.zeros(n_test)
parameter_max = np.zeros(n_test)

# For each leaky rate values
index = 0
for leaky_rate in np.linspace(0.001, 0.002, n_test):
    # Log
    print(u"Leaky rate : {}".format(leaky_rate))

    # Samples average
    samples_average = np.array([])

    # For each samples
    for n in range(n_samples):
        # Create W matrix
        w = etnn.ESNCell.generate_w(reservoir_size, w_sparsity)

        # Truth and prediction
        y_true = np.array([])
        y_pred = np.array([])

        # Sample average
        single_sample_average = np.array([])

        # For each problem
        for problem in np.arange(1, 3):
            # Author identification training dataset
            pan18loader_training = torch.utils.data.DataLoader(
                dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transformer, problem=problem, lang=args.lang),
                batch_size=1, shuffle=True
            )

            # Author identification test dataset
            pan18loader_test = torch.utils.data.DataLoader(
                dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transformer, problem=problem, train=False, lang=args.lang),
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
            esn = etnn.LiESN(
                input_dim=transformer.transforms[1].input_dim,
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

            # Precision and recall
            precision = float(np.sum(y_true == y_pred)) / float(y_pred.shape[0])
            recall = recall_score(y_true, y_pred, average='macro')

            # F1
            f1_score = 2.0 * ((precision * recall) / (precision + recall))

            # Show accuracy
            # print(u"\tProblem {}, Precision : {}, Recall : {}, F1 _ {}".format(problem, precision, recall, f1_score))

            # Save result
            single_sample_average = np.append(single_sample_average, [f1_score])

            # Reset ESN
            esn.reset()
        # end for

        # Save results
        samples_average = np.append(samples_average, [np.average(single_sample_average)])
    # end for

    # Show result
    print(u"\tMacro average F1 score : {} (max {})".format(np.average(samples_average), np.max(samples_average)))

    # Save results
    parameter_averages[index] = np.average(samples_average)
    parameter_averages[index] = np.max(samples_average)
    index += 1
# end for

# Show
print(parameter_averages)
print(parameter_max)
print(np.max(parameter_max))

# Show results
plt.plot(parameter_averages)
plt.show()
