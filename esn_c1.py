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
from echotorch.transforms import text
from torch.autograd import Variable
import echotorch.utils

# Experiment settings
reservoir_size = 1000
spectral_radius = 0.95
input_sparsity = 0.1
w_sparsity = 0.1
input_scaling = 0.5
leak_rate = 0.01


def zero_vector(char_idx):
    return torch.zeros(10)
# end zero_vector


# Character + Embeddings
transform = text.Compose(transforms=[
    text.Character(),
    text.Embedding(embedding=zero_vector, size=10)
])

# Author identification training dataset
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=1),
    batch_size=1, shuffle=True)

# Author identification test dataset
pan18loader_test = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=1,
                                        train=False),
    batch_size=1, shuffle=True)

# Authors
author_to_idx = dict()
for idx, author in enumerate(pan18loader_training.dataset.authors):
    author_to_idx[author] = idx
# end for

# Number of authors
n_authors = len(author_to_idx)

# ESN cell
esn = etnn.LiESN(
    input_dim=transform.input_dim,
    hidden_dim=reservoir_size,
    output_dim=n_authors,
    spectral_radius=spectral_radius,
    sparsity=input_sparsity,
    input_scaling=input_scaling,
    w_sparsity=w_sparsity,
    learning_algo='inv',
    leaky_rate=leak_rate
)

# Get training data for this fold
for i, data in enumerate(pan18loader_training):
    # Inputs and labels
    inputs, labels = data

    # Create time labels
    author_id = author_to_idx[labels[0]]
    tag_vector = torch.zeros(inputs.size(1), n_authors)
    tag_vector[:, author_id] = 1.0

    # To variable
    inputs, time_labels = Variable(inputs), Variable(tag_vector)

    # Accumulate xTx and xTy
    # esn(inputs, time_labels)
# end for

# Finalize training
# esn.finalize()

# Counters
successes = 0.0
count = 0.0

# Get test data
for i, data in enumerate(pan18loader_test):
    # Inputs and labels
    inputs, labels = data

    # Author id
    author_id = author_to_idx[labels[0]]

    # Predict
    """y_predicted = esn(inputs)

    # Normalized
    y_predicted -= torch.min(y_predicted)
    y_predicted /= torch.max(y_predicted) - torch.min(y_predicted)

    # Sum to one
    sums = torch.sum(y_predicted, dim=2)
    for t in range(y_predicted.size(1)):
        y_predicted[0, t, :] = y_predicted[0, t, :] / sum[0, t]
    # end for

    # Max average through time
    y_predicted = echotorch.utils.max_average_through_time(y_predicted, dim=1)

    # Compare
    if torch.equal(y_predicted, labels):
        successes += 1.0
    # end if"""
# end for

print(u"Accuracy : {}".format(successes / count))
