# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchvision import transforms


################
# Settings
################

# Parameters
spectral_radius = 0.95
input_sparsity = 0.1
w_sparsity = 0.1
input_scaling = 0.5
n_test = 20
n_samples = 5000
leaky_rate = 0.1
reservoir_size = 400

# Settings
lang_models = {
    'en': 'en_vectors_web_lg',
    'fr': 'fr_core_news_md',
    'it': '~/Projets/TURING/Datasets/fasttext/wiki.it/wiki.it.vec',
    'sp': '~/Projets/TURING/Datasets/fasttext/wiki.es/wiki.es.vec',
    'pl': '~/Projets/TURING/Datasets/fasttext/wiki.pl/wiki.pl.vec'
}

lang_spacy_models = {
    'en': 'en_vectors_web_lg',
    'fr': 'fr_core_news_md',
    'it': 'nltk',
    'sp': 'es_core_news_md',
    'pl': 'nltk'
}


lang_models_dim = {
    'en': 300,
    'fr': 300,
    'it': 300,
    'sp': 300,
    'pl': 300,
}


lang_models_lang = {
    'en': 'en',
    'fr': 'french',
    'it': 'italian',
    'sp': 'spanish',
    'pl': 'polish',
}

