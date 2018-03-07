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
from echotorch.transforms import text
import random
from torch.autograd import Variable

# Experience parameter
window_size = 500
batch_size = 64
sample_batch = 4
epoch_batches = 10
max_epoch = 1

# Author identification training dataset
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=text.GloveVector(), problem=1),
    batch_size=1, shuffle=True)

# Author identification test dataset
pan18loader_test = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=text.GloveVector(), problem=1,
                                        train=False),
    batch_size=1, shuffle=True)

# Authors
author_to_idx = dict()
for idx, author in enumerate(pan18loader_training.dataset.authors):
    author_to_idx[author] = idx
# end for

# Number of authors
n_authors = len(author_to_idx)

# Total training data
training_data = list()
training_labels = list()

# Get training data
for i, data in enumerate(pan18loader_training):
    # Inputs and labels
    inputs, labels = data

    # Add
    training_data.append(inputs)
    training_labels.append(labels)
# end for

# Number of samples
n_samples = len(training_labels)

# For each iteration
for epoch in range(max_epoch):
    # For each batch
    for b in range(epoch_batches):
        # Batch labels
        batch_labels = torch.LongTensor(batch_size)

        # Get samples for the batch
        for i in range(batch_size):
            # Random sample and position
            random_sample = random.randint(0, n_samples-1)
            random_sample_size = training_data[random_sample].size(1)
            random_position = random.randint(0, random_sample_size-window_size-1)
            sample = training_data[random_sample]

            # Get sequence
            random_sequence = sample[:, random_position:random_position+window_size]

            # Append
            if i == 0:
                batch = random_sequence
            else:
                batch = torch.cat((batch, random_sequence), dim=0)
            # end if

            # Label
            batch_labels[i] = author_to_idx[training_labels[random_sample][0]]
        # end for

        # To variable
        inputs, labels = Variable(batch), Variable(batch_labels)
    # end for

    # Counter
    success = 0.0
    count = 0.0

    # For each test sample
    for i, data in enumerate(pan18loader_test):
        # Inputs and labels
        inputs, labels = data

        # Labels as tensor
        labels = [author_to_idx[l] for l in labels]
        labels = torch.LongTensor(labels)

        # Sample length
        sample_length = inputs.size(1)

        # Authors predictions
        author_predictions = torch.zeros(sample_length-window_size, n_authors)

        # Sliding window
        for pos in range(0, sample_length-window_size):
            # Window data
            window_data = inputs[0, pos:pos+window_size]

            # Classify
            author_predictions[pos] = torch.zeros(1, n_authors)
            author_predictions[pos, 0] = 1.0
        # end for

        # Max average probability through time
        average_probs = torch.mean(author_predictions, dim=0)

        # Predicted author
        _, indice = torch.max(average_probs, dim=0)

        # Compare

        if torch.equal(labels, indice):
            success += 1.0
        # end if
        count += 1.0
    # end for

    # Display accuracy
    print(u"Test accuracy : {}".format(success / count * 100.0))
# end for
