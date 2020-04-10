# CS114 Spring 2020 Programming Assignment 4
# Part-of-speech Tagging with Hidden Markov Models

import os
import numpy as np
from collections import defaultdict

class POSTagger():

    def __init__(self):
        self.pos_dict = {}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.UNK = '<UNK>'
        self.k = 1

    '''
    Trains a supervised hidden Markov model on a training set.
    self.initial[POS] = log(P(the initial tag is POS))
    self.transition[POS1][POS2] =
    log(P(the current tag is POS2|the previous tag is POS1))
    self.emission[POS][word] =
    log(P(the current word is word|the current tag is POS))
    '''
    def train(self, train_set):
<<<<<<< HEAD
        # Construct index dicts
=======
        self.initial = np.zeros([0]) # Initial is a 1-D array
        self.transition = np.zeros([0, 0]) # Transition is a 2-D matrix
        self.emission = np.zeros([0, 0]) # Emission is a 2-D matrix

        # iterate over training documents
>>>>>>> ce7a3ab1bc7583417d6572b4ced46ba3b1a2888f
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # Split the document into sentences
                    lines = f.readlines()
                for line in lines:
<<<<<<< HEAD
=======
                    prev_tag = None
>>>>>>> ce7a3ab1bc7583417d6572b4ced46ba3b1a2888f
                    tagged_tokens = line.split()
                    for pair in tagged_tokens:
                        # Split the pair by the last occurence of '/'
                        token, tag = pair.rsplit('/', 1)
<<<<<<< HEAD
                        # Add indices for the current token and tag, if has not been added before
                        if not token in self.word_dict:
                            self.word_dict[token] = len(self.word_dict)
                        if not tag in self.pos_dict:
                            self.pos_dict[tag] = len(self.pos_dict)
        self.word_dict[self.UNK] = len(self.word_dict) # Add UNK
        # Initialize counters
        num_of_tags, num_of_tokens = len(self.pos_dict), len(self.word_dict)
        self.initial = np.zeros([num_of_tags])
        self.transition = np.zeros([num_of_tags, num_of_tags])
        self.emission = np.zeros([num_of_tags, num_of_tokens])
        # Construct counters
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    lines = f.readlines()
                for line in lines:
                    tagged_tokens = line.split()
                    prev_tag = None
                    for pair in tagged_tokens:
                        token, tag = pair.rsplit('/', 1)
                        # Count emission, based on the pair
                        cur_tag_ind = self.pos_dict[tag]
                        cur_token_ind = self.word_dict[token]
                        self.emission[cur_tag_ind][cur_token_ind] += 1
                        # If prev is None, add to self.initial. Otherwise Add to self.transition, based on the previous tag.
                        if prev_tag is None:
                            self.initial[cur_tag_ind] += 1
                        else:
                            prev_tag_ind = self.pos_dict[prev_tag]
                            self.transition[prev_tag_ind][cur_tag_ind] += 1
                        prev_tag = tag
        # Set all unk-related count to 1
        for row in self.emission:
            row[self.word_dict[self.UNK]] = 1
        # Add-k smoothing
        self.initial += 1
        self.emission += 1
        self.transition += 1
        # Compute probabilities 
        self.initial = self.initial / self.initial.sum()
        self.emission = self.emission / self.emission.sum(axis = 1, keepdims = True)
        self.transition = self.transition / self.transition.sum(axis = 1, keepdims = True)
        # Validate probabilities sum to one
        self.__validate_probabilities()
=======
>>>>>>> ce7a3ab1bc7583417d6572b4ced46ba3b1a2888f

                        # Add indices for the current token and tag, if has not been added before
                        # When a new word/tag appears, we also need to add a new row/col to matrices
                        if not token in self.word_dict:
                            self.word_dict[token] = len(self.word_dict)
                            self.emission = np.append(self.emission, np.zeros([self.emission.shape[0], 1]), axis = 1) # Add a new col to emission
                        if not tag in self.pos_dict:
                            self.pos_dict[tag] = len(self.pos_dict)
                            self.initial = np.append(self.initial, 0) # Added a place holder for new tag
                            # Add a new col and row in transition
                            self.transition = np.append(self.transition, np.zeros([1, self.transition.shape[1]]), axis = 0)
                            self.transition = np.append(self.transition, np.zeros([self.transition.shape[0], 1]), axis = 1)
                            self.emission = np.append(self.emission, np.zeros([1, self.emission.shape[1]]), axis = 0) # Add a new row in emission
                        
                        # Count emission, based on the pair
                        cur_tag_ind = self.pos_dict[tag]
                        cur_token_ind = self.word_dict[token]
                        self.emission[cur_tag_ind][cur_token_ind] += 1

                        # If prev is None, add to initial. Otherwise Add to Transition, based on the previous tag.
                        if prev_tag is None:
                            self.initial[cur_tag_ind] += 1
                        else:
                            prev_tag_ind = self.pos_dict[prev_tag]
                            self.transition[prev_tag_ind][cur_tag_ind] += 1
                        prev_tag = tag
        # Add a col for <UNK> in self.emission
        new_col = np.ones([self.emission.shape[0], 1])
        self.emission = np.append(self.emission, new_col, axis = 1)
        # Add-k smoothing
        self.initial += 1
        self.emission += 1
        self.transition += 1
        # Compute probabilities 
        self.initial = self.initial / self.initial.sum()
        self.emission = self.emission / self.emission.sum(axis = 1, keepdims = True)
        self.transition = self.transition / self.transition.sum(axis = 1, keepdims = True)
        # Validate probabilities sum to one
        self.__validate_probabilities()

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        v = None
        backpointer = None
        # initialization step
        # recursion step
        # termination step
        best_path = []
        return best_path

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of POS tags such that:
    results[sentence_id]['correct'] = correct sequence of POS tags
    results[sentence_id]['predicted'] = predicted sequence of POS tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    pass
        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        accuracy = 0.0
        return accuracy

    def __validate_probabilities(self):
        tor = 1.e-10
        if abs(self.initial.sum() - 1) > tor:
            print("Initial probabilities do not sum to one")
        for row in self.emission:
            if abs(row.sum() - 1) > tor:
                print("Emission probabilities do not sum to one")
        for row in self.transition:
            if abs(row.sum() - 1) > tor:
                print("Transition probabilities do not sum to one")

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('train')
    results = pos.test('dev')
    # pos.train('train_small')
    # results = pos.test('test_small')
    print('Accuracy:', pos.evaluate(results))
