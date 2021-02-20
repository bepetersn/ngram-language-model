
# CSCI 3832, Spring 2021, CU Boulder

from collections import Counter
from collections import defaultdict
import numpy as np
import re
from matplotlib import pyplot as plt
import sys


class LanguageModel:

    def __init__(self, ngram_order, is_laplace_smoothing):
        self.ngram = ngram_order
        self.is_laplace_smoothing = is_laplace_smoothing
        self.ngram_counts = {}
        # ...

    # TODO: comment your functions
    # Add function comments and comment code in line that is doing
    # complex things
    def train(self, training_file_path):
        """Train stuff

        Args:
            training_file_path (string): The path to a file
                                         with relevant training data
        """

        print('training done')
        print('num of unique {}-grams :={}'.format())#todo: put the proper variables
        # no return

    def score(self, sentence):
        """Return the probability of the sentence

        Args:
            sentence (string): Sentence consisting of space-delimited 
                               words.
        """
        return

    def getPerplexity(self, filename):
        """Return perplexity of the file

        Args:
            filename (string): A filepath for testing against

        Returns:
            [type]: [description]
        """

        print('perplexity using {}-grams :={}'.format())#todo: put the proper variables
        return perplexity

    def generate(self, num_sentences):
        """Generate a list of sentences using Shannon's method

        Args:
            num_sentences (int): The number of sentences to
                                 generate

        Returns:
            list[string]: A list of sentences containing space-delimited words
        """
        randomsentences=[]

        return randomsentences


if __name__ == '__main__':

    # ADDED
    if len(sys.argv) != 3:
        print("Usage:", "python hw2_lm.py berp-training.txt hw2-test.txt ")
        sys.exit(1)

    trainingFilePath = sys.argv[1]
    testFilePath = sys.argv[2]

    # Generate probability for each sentence
    # in the test set using the training data

    # Generate 100 sentences

