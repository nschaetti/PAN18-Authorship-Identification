# -*- coding: utf-8 -*-
#

# Imports
from torch.utils.data.dataset import Dataset
import urllib
import os
import zipfile
import json
import codecs


# Author identification data set
class AuthorIdentificationDataset(Dataset):
    """
    Author identification dataset
    """

    # Constructor
    def __init__(self, root='./data', download=True, lang='en', problem=0, train=True, transform=None):
        """
        Constructor
        :param root:
        :param download:
        :param lang:
        :param problem:
        """
        # Properties
        self.root = root
        self.lang = lang
        self.problem = problem
        self.train = train
        self.transform = transform
        self.authors = list()

        # List of text
        self.texts = list()

        # Create directory
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Download the data set
        if download and not os.path.exists(os.path.join(self.root, "collection-info.json")):
            self._download()
        # end if

        # Generate data set
        self._load()
    # end if

    ##############################################
    # OVERRIDE
    ##############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.texts)
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        # Current file
        text_path, author_name = self.texts[item]

        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # No tab and return
        text_content = text_content.replace(u"\n", u" ").replace(u"\t", u" ")

        # Transformed
        transformed, transformed_size = self.transform(text_content)

        # Transform
        if self.transform is not None:
            return transformed, author_name
        else:
            return text_content, author_name
        # end if
    # end __getitem__

    ################################################
    # PRIVATE
    ################################################

    # Create the root directory
    def _create_root(self):
        """
        Create the root directory
        :return:
        """
        os.mkdir(self.root)
    # end _create_root

    # Download the dataset
    def _download(self):
        """
        Download the dataset
        :return:
        """
        # Path to zip file
        path_to_zip = os.path.join(self.root, "pan18-author-identification.zip")

        # Download
        urllib.urlretrieve("http://www.nilsschaetti.com/datasets/pan18-author-identification.zip", path_to_zip)

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    # Load dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        collection_info = json.load(open(os.path.join(self.root, "collection-info.json"), 'r'))

        # Problem index
        problem_index = 0

        # For each entry
        for problem in collection_info:
            if problem['language'] == self.lang and problem_index == self.problem:
                # Problem name
                problem_name = problem['problem-name']

                # Load problem info
                problem_info = json.load(open(os.path.join(self.root, problem_name, "problem-info.json")))

                # Info
                unknown_folder = problem_info['unknown-folder']
                candidate_authors = problem_info['candidate-authors']

                # Load texts
                if self.train:
                    # Load candidate texts
                    for candidate_author in candidate_authors:
                        # Author name
                        author_name = candidate_author['author-name']

                        # Add  to authors
                        if author_name not in self.authors:
                            self.authors.append(author_name)
                        # end if

                        # For each text
                        for file_path in os.listdir(os.path.join(self.root, problem_name, author_name)):
                            # Add with author name
                            self.texts.append((os.path.join(self.root, problem_name, author_name, file_path), author_name))
                        # end for
                    # end for
                else:
                    # Ground truth
                    ground_truths = json.load(open(os.path.join(self.root, problem_name, "ground-truth.json")))

                    # For each test files
                    for ground_truth in ground_truths['ground_truth']:
                        # Add with author name
                        self.texts.append((os.path.join(self.root, problem_name, unknown_folder, ground_truth['unknown-text']), ground_truth['true-author']))
                    # end for
                # end if
            elif problem['language'] == self.lang:
                problem_index += 1
            # end if
        # end for
    # end _load

# end AuthorIdentificationDataset
