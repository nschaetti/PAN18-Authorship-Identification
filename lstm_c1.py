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

# Experience parameter
max_epoch = 1

# Author identification training dataset
pan18loader_training = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=text.Character(), problem=1),
    batch_size=1, shuffle=True)

# Author identification test dataset
pan18loader_test = torch.utils.data.DataLoader(
    dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=text.Character(), problem=1,
                                        train=False),
    batch_size=1, shuffle=True)

# Authors
author_to_idx = dict()
for idx, author in enumerate(pan18loader_training.dataset.authors):
    author_to_idx[author] = idx
# end for

# Number of authors
n_authors = len(author_to_idx)

# For each iteration
for epoch in range(max_epoch):
    # Get training data
    for i, data in enumerate(pan18loader_training):
        # Inputs and labels
        inputs, labels = data

        # Author ID
        author_id = author_to_idx[labels[0]]

        print(inputs.size())
        print(author_id)
    # end for
# end for

# Counters
successes = 0.0
count = 0.0

# Get test data
for i, data in enumerate(pan18loader_test):
    # Inputs and labels
    inputs, labels = data

    # Author id
    author_id = author_to_idx[labels[0]]
# end for

print(u"Accuracy : {}".format(successes / count))

