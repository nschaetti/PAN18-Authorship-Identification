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
import numpy as np
from modules import CNNC
import torch.nn as nn
from torch import optim

# Experience parameter
voc_size = 77
window_size = 500
batch_size = 32
sample_batch = 4
epoch_batches = 10
max_epoch = 1
stride = 10

# Transformer
transform = text.Character()

# Author identification training dataset
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=0, train=True),
    batch_size=1, shuffle=True)

# Author identification test dataset
pan18loader_test = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=0,
                                        train=False),
    batch_size=1, shuffle=True)

# Authors
author_to_idx = dict()
for idx, author in enumerate(pan18loader_training.dataset.authors):
    author_to_idx[author] = idx
# end for

# Number of authors
n_authors = len(author_to_idx)

# Training batches
training_samples = list()

# Get training data
for i, data in enumerate(pan18loader_training):
    # Inputs and labels
    inputs, labels = data

    # Sliding window
    for s in np.arange(0, inputs.size(1)-window_size, stride):
        # Get windowed sample
        window_sample = inputs[:, s:s+window_size]

        # Add to training samples
        training_samples.append((window_sample, author_to_idx[labels[0]]))
    # end for
# end for

# Number of samples
n_training_samples = len(training_samples)

# Shuffle the list
random.shuffle(training_samples)

# Model
net = CNNC(vocab_size=voc_size, n_classes=10)

# Loss function
loss_function = nn.NLLLoss()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001)

# For each iteration
for epoch in range(max_epoch):
    # For each batch
    for b in np.arange(0, n_training_samples-batch_size, batch_size):
        # Batch
        batch = torch.LongTensor(batch_size, window_size)
        labels = torch.LongTensor(batch_size)

        # For each sample
        index = 0
        for s in np.arange(b, b+batch_size):
            batch[index] = training_samples[s][0][0]
            labels[index] = training_samples[s][1]
            index += 1
        # end for

        # To variable
        inputs, labels = Variable(batch), Variable(labels)

        # Zero grad
        net.zero_grad()

        # Compute output
        log_probs = net(inputs)

        # Loss
        loss = loss_function(log_probs, labels)

        # Backward and step
        loss.backward()
        optimizer.step()
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

        # Average probability over time
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
