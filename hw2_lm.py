
# CSCI 3832, Spring 2021, CU Boulder

from collections import Counter
from collections import defaultdict
from funcy import post_processing, collecting
from matplotlib import pyplot as plt
import itertools as it
import numpy as np
from numpy.random import choice
import re
import sys
import math

# TODO: Somewhere, plot a histogram of unigram word counts, in desc. order:
#       x-axis: words; y-axis: count

UNIGRAM_OUT_FILE = "hw2-unigram-out.txt" 
BIGRAM_OUT_FILE = "hw2-bigram-out.txt"
TRIGRAM_OUT_FILE = "hw2-trigram-out.txt"
GENERATED_UNIGRAM = "hw2-unigram-generated.txt"
GENERATED_BIGRAM = "hw2-bigram-generated.txt"
GENERATED_TRIGRAM = "hw2-trigram-generated.txt"


def zeroDefaultDictFactory(d=None):
    return {} if d is None else defaultdict(lambda: 0, d)


class LanguageModel:

    COUNT_TO_BE_BAD = 1
    UNK_TOKEN = "<unk>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        self.ngram_size = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing

        self.bad_word_list = []
        self.vocab = []
        
        self.ngram_probabilities = zeroDefaultDictFactory()
        if self.ngram_size > 1:
            # NOTE: below, "nlo" = next lowest order
            self.nlo_ngram_probabilities = zeroDefaultDictFactory()
        else:
            self.nlo_ngram_probabilities = defaultdict(lambda: 1) 
        print('initialized')

    def train(self, training_file_path):
        """Train our model on a file containing
           one sentence per line delimited by 
           `self.ngram_size` number of `START_TOKEN`
           and `END_TOKEN` tokens at their beginning
           and end respectively.

           After this has been run, self.bad_word_list, self.vocab, 
           self.ngram_probabilities, and possibly self.nlo_gram_probabilities
           will be set.

        Args:
            training_file_path (string): The path to a file
                                         with relevant training data
        """
        token_counts = Counter()
        ngram_counts = Counter()
        ngram_total_count = 0

        # If we are looking at bigrams or higher orders, we 
        # track the next lowest order (nlo) counts as well,
        # over all data to eventually generate probabilities
        if self.ngram_size > 1:
            nlo_ngram_counts = Counter()
            nlo_ngram_total_count = 0
    
        with open(training_file_path) as f:
            lines = f.readlines()

        # Find the token frequencies
        for line in lines:
            line_tokens = line.split()
            token_counts.update(line_tokens) 
        
        # Choose and store a vocab / "bad" word 
        # list using the frequencies
        self.bad_word_list = self._take_bad_words(token_counts)
        self.vocab         = self._filter_out_bad_words(token_counts)

        # Train on each line
        for line in lines:
            line_tokens = self._replace_bad_words(line.split())
            ngram_total_count += self._train_on_line(
                    self.ngram_size, line_tokens, ngram_counts)
            if self.ngram_size > 1:
                nlo_ngram_total_count += self._train_on_line(
                        self.ngram_size-1, line_tokens, nlo_ngram_counts)

        # Derive and store the probabilities for each 
        # n-gram and nlo ngram, if appropriate
        self.ngram_probabilities = \
            self._derive_probabilities(
                ngram_total_count, ngram_counts)
        if self.ngram_size > 1:
            self.nlo_ngram_probabilities = \
                self._derive_probabilities(
                        nlo_ngram_total_count, nlo_ngram_counts)

        print('training done')
        print('num of unique {}-grams :={}'
              ''.format(self.ngram_size, len(ngram_counts.keys())))
    
    def _replace_bad_words(self, tokens):
        """Return a transformation of tokens
           with "bad" words replaced by UNK_TOKEN"""
        return [self.UNK_TOKEN if t in self.bad_word_list 
                               else t for t in tokens]

    def _take_bad_words(self, token_counts):
        return [k for (k, v) in token_counts.items() if v <= self.COUNT_TO_BE_BAD]

    def _filter_out_bad_words(self, token_counts):
        return [k for (k, v) in token_counts.items() if v > self.COUNT_TO_BE_BAD]

    def _train_on_line(self, ngram_size, tokens, ngram_counts):
        """Update our map of ngram_counts with all the ngrams to 
           which the given tokens map, using the given `ngram`.
           Return the number of ngrams encountered.

        Args:
            ngram_size (int): The size of the length of the n-grams we 
                         are going to use to model our training data.
            tokens (list): A list of tokens corresponding to one
                           sentence / line on which to train
            ngram_counts (Counter): A mapping between each n-gram
                                    and the number of times it has
                                    occurred so far in training 
                                    (we mutate this data structure)
        Returns: 
            int: The total number of ngrams to which the given 
                 tokens mapped
        
        """

        ngrams = self._create_ngrams(ngram_size, tokens)
        ngram_counts.update(ngrams)
        return len(ngrams)

    @post_processing(zeroDefaultDictFactory)
    def _derive_probabilities(self, ngram_total_count, ngram_counts):
        """Derive probabilities of each ngram from counts and total

        Args:
            ngram_total_count (int): The sum of the number of 
                                     occurences of each n-gram
                                     in training
            ngram_counts (Counter): A mapping between each n-gram
                                    and the number of times it 
                                    occurred in training
        Yields:
            dict: A mapping between each n-gram and
                  its probability among all n-grams 
                  in training
        """
        for ngram, count in ngram_counts.items():
            yield (ngram, (count / ngram_total_count))

    @staticmethod
    def _create_ngrams(ngram_size, tokens):
        """Turn a list of tokens into a list of n-grams 
        of the given ngram_size. 
    
        Args:
            tokens (list): A list of tokens to be turned
                           into corresponding n-grams.
            ngram_size (int): The size of the length of the 
                              n-grams to create

        Returns:
            list[tuple]: A list of tuples of size `ngram`
                         corresponding to `tokens`. Given, 
                         e.g., ["I", "am", "Sam"] and 
                         ngram size 2, return 
                         [("I", "am"), ("am", "Sam")]
        """
        ngrams = []
        for i in range(ngram_size-1, len(tokens)):
            ngram = tuple(tokens[i-ngram_size+1:i+1])
            ngrams.append(ngram)
        return ngrams

    def score(self, sentence):
        """Return the probability of the sentence

        Args:
            sentence (string): Sentence consisting of space-delimited 
                               words.
        Returns: 
            float: Log-space probability of this sentence given 
                   our model's training.
        """
        tokens = sentence.split()
        log_probability = 0 
        # Replace unknown words with UNK_TOKEN
        for i, token in enumerate(tokens):
            if token not in self.vocab:
                tokens[i] = self.UNK_TOKEN
        
        ngrams = self._create_ngrams(self.ngram_size, tokens)
        for ngram in ngrams:
            log_probability += self._score_ngram(ngram)

        return log_probability
    
    def _score_ngram(self, ngram):
        # Below: backoff from, e.g., ("I", "am", "Sam") to ("I", "am"),
        #        the effective context, with ngram[:2] 
        nlo_ngram = ngram[:self.ngram_size-1]
        ngram_prob = self.ngram_probabilities[ngram]
        nlo_ngram_prob = self.nlo_ngram_probabilities[nlo_ngram]
        
        if self.is_laplace_smoothing:
            ngram_prob += 1
            nlo_ngram_prob += len(self.ngram_probabilities.keys())

        return math.log(ngram_prob / nlo_ngram_prob, 2)


    def getPerplexity(self, filename):
        """Return perplexity of the file

        Args:
            filename (string): A filepath for testing against

        Returns:
            float: Perplexity of the file using the training data
                   and a certain size of n-gram language model 
        """

        with open(filename) as f:
            content = ' '.join(f.readlines())
        tokens = content.split()
        num_tokens = len(tokens)

        # Replace unknown words with UNK_TOKEN
        for i, token in enumerate(tokens):
            if token not in self.vocab:
                tokens[i] = self.UNK_TOKEN

        # This is basically the equation for perplexity in log space
        perplexity = self.score(''.join(tokens))
        perplexity *= -(1 / num_tokens)
        perplexity = math.exp(perplexity)

        print('perplexity using {}-grams :={}'
              ''.format(self.ngram_size, perplexity))
        return perplexity

    @collecting
    def generate(self, num_sentences):
        """Generate a list of sentences using Shannon's method

        Args:
            num_sentences (int): The number of sentences to
                                 generate using our approach

        Returns:
            list[string]: A list of sentences containing 
                          space-delimited words
        """
        ngram_probabilities = self.ngram_probabilities.copy()
        # Below: We don't want any start tokens showing up;
        #        This is not a concern for anything other than 
        #        unigrams, as we will never have training bridging 
        #        from any context to selecting a start token
        if self.ngram_size == 1:
            del ngram_probabilities[(self.START_TOKEN, )] 
        ngrams = list(ngram_probabilities.keys())

        for i in range(num_sentences):
            yield self._generate_sentence(ngrams, ngram_probabilities)

    def _generate_sentence(self, ngrams, ngram_probabilities):

        # Build a sentence
        prev = self.START_TOKEN
        # For bigrams and higher, we need at least n-1 start tokens
        if self.ngram_size > 1:
            sentence = [prev] * (self.ngram_size-1)
        else:
            sentence = [prev]
        
        while True:

            # Choose next word based on probabilities
            # of those starting with previously selected,
            # except if we are using a unigram model    
            if self.ngram_size > 1:
                ngrams_starting_with_prev = \
                    [n for n in ngrams if n[0] == prev] 
            else:
                ngrams_starting_with_prev = ngrams

            probs_for_ngrams_starting_w_prev = \
                [ngram_probabilities[n] \
                    for n in ngrams_starting_with_prev]
            next_ngram = self._choose_using_distribution(
                ngrams_starting_with_prev,
                probs_for_ngrams_starting_w_prev
            )
            next_token = next_ngram[-1]
            sentence.append(next_token)
            prev = next_token
            if next_token == self.END_TOKEN:
                break
        
        # For bigrams and higher, need n-1 end tokens;
        # Ensure only that many are present 
        if self.ngram_size > 1:
            sentence = sentence[:-1] 
            sentence.extend([self.END_TOKEN] * (self.ngram_size-1))

        return sentence

    @staticmethod
    def _choose_using_distribution(choices, probabilities):
        probabilities = np.array(probabilities)
        # Below: Normalize, so numpy doesn't complain about 
        #        summing to 1
        probabilities /= probabilities.sum()
        # Below: Choose an ngram from those available
        #        using the probability distribution;
        #        use len() because it looks like a 2d array,
        #        and [0] because numpy preserves array
        next_ngram_index = choice(
            len(choices), size=1, p=probabilities)[0]
        return choices[next_ngram_index]


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage:", "python hw2_lm.py berp-training.txt hw2-test.txt ")
        sys.exit(1)

    trainingFilePath = sys.argv[1]
    testFilePath = sys.argv[2]

    lm1 = LanguageModel(1, False)
    lm1.train(trainingFilePath)

    lm2 = LanguageModel(2, True)
    lm2.train(trainingFilePath)

    lm3 = LanguageModel(3, True)
    lm3.train(trainingFilePath)

    # Generate probability for each sentence
    # in the test set using the training data...
    with open(testFilePath) as f:
        lines = f.readlines()
    
    # ... and a unigram model
    with open(UNIGRAM_OUT_FILE, 'w') as f:
        for line in lines:
            f.write(str(lm1.score(line.strip())) + "\n")

    # Generate 100 sentences using unigram model
    with open(GENERATED_UNIGRAM, 'w') as f:
        for sentence in lm1.generate(100):
            f.write(' '.join(sentence) + "\n")

    # Generate probability for each sentence 
    # in the test using training data and bigram model
    with open(BIGRAM_OUT_FILE, 'w') as f:
        for line in lines:
            f.write(str(lm2.score(line.strip())) + "\n")

    # Generate 100 sentences using bigram model
    with open(GENERATED_BIGRAM, 'w') as f:
        for sentence in lm2.generate(100):
            f.write(' '.join(sentence) + "\n")

    # Generate 100 sentences using bigram model
    with open(GENERATED_TRIGRAM, 'w') as f:
        for sentence in lm3.generate(100):
            f.write(' '.join(sentence) + "\n")

    # Generate probability for each sentence 
    # in the test using training data and bigram model
    with open(TRIGRAM_OUT_FILE, 'w') as f:
        for line in lines:
            f.write(str(lm3.score(line.strip())) + "\n")

            

