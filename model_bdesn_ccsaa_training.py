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
import torch.nn as nn
import echotorch.nn as etnn
from torch.autograd import Variable
import echotorch.utils
import torchlanguage.transforms as transforms
import torchlanguage.models
from torch import optim
import tools.functions
import tools.settings
import numpy as np
import os

# Experiment settings
reservoir_size = 300
spectral_radius = 0.95
input_sparsity = 0.1
w_sparsity = 0.1
input_scaling = 0.5
n_test = 10
n_samples = 2
n_epoch = 100
text_length = 20

# Argument
args = tools.functions.argument_parser_training_model()

# Transforms
transform = transforms.Compose([
    transforms.Character(),
    transforms.ToIndex(start_ix=0),
    transforms.MaxIndex(max_id=83),
    transforms.ToNGram(n=text_length, overlapse=True),
    transforms.Reshape((-1, 20))
])

# Author identification training dataset
dataset_train = dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=1, lang='en')

# Author identification test dataset
dataset_valid = dataset.AuthorIdentificationDataset(root="./data/", download=True, transform=transform, problem=1, train=False, lang='en')

# Cross validation
dataloader_train = torch.utils.data.DataLoader(torchlanguage.utils.CrossValidation(dataset_train), batch_size=1, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(torchlanguage.utils.CrossValidation(dataset_valid, train=False), batch_size=1, shuffle=True)

# Author to idx
author_to_ix = dict()
for idx, author in enumerate(dataset_train.authors):
    author_to_ix[author] = idx
# end for

# Model
model = torchlanguage.models.CCSAA(text_length=text_length, vocab_size=84, embedding_dim=50, n_classes=20)
if args.cuda:
    model.cuda()
# end if

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Total losses
training_loss = 0.0
test_loss = 0.0

# Get training data for this fold
for i, data in enumerate(dataloader_train):
    # Inputs and labels
    inputs, labels = data

    # Reshape
    inputs = inputs.view(-1, text_length)

    # Outputs
    outputs = torch.LongTensor(inputs.size(0)).fill_(author_to_ix[labels[0]])

    # To variable
    inputs, outputs = Variable(inputs), Variable(outputs)
    if args.cuda:
        inputs, outputs = inputs.cuda(), outputs.cuda()
    # end if

    # Zero grad
    model.zero_grad()

    # Compute output
    log_probs = model(inputs)

    # Loss
    loss = loss_function(log_probs, outputs)

    # Backward and step
    loss.backward()
    optimizer.step()

    # Add
    training_loss += loss.data[0]
# end for

# Counters
total = 0.0
success = 0.0

# Get test data
for i, data in enumerate(dataloader_valid):
    # Inputs and labels
    inputs, labels = data

    # Reshape
    inputs = inputs.view(-1, text_length)

    # Outputs
    outputs = torch.LongTensor(inputs.size(0)).fill_(author_to_ix[labels[0]])

    # To variable
    inputs, outputs = Variable(inputs), Variable(outputs)
    if args.cuda:
        inputs, outputs = inputs.cuda(), outputs.cuda()
    # end if

    # Forward
    model_outputs = model(inputs)
    loss = loss_function(model_outputs, outputs)

    # Take the max as predicted
    _, predicted = torch.max(model_outputs.data, 1)

    # Add to correctly classified word
    success += (predicted == outputs.data).sum()
    total += predicted.size(0)

    # Add loss
    test_loss += loss.data[0]
# end for

# Accuracy
accuracy = success / total * 100.0

# Print and save loss
print(u"Training loss {}, test loss {}, accuracy {}".format(training_loss, test_loss, accuracy))

# Save model
torch.save(model.state_dict(), open(u"./cnn_character_extractor.pth", 'wb'))
torch.save(transform.transforms[1].token_to_ix, open(u"./cnn_character_extractor.voc.pth", 'wb'))
