# CS114 Spring 2020 Programming Assignment 4
# Part-of-speech Tagging with Hidden Markov Models

import os
import numpy as np
from collections import defaultdict

class POSTagger():

    def __init__(self, k):
        self.pos_dict = {}
        self.reverse_pos_dict = {}
        self.word_dict = {}
        self.initial = None
        self.transition = None
        self.emission = None
        self.UNK = '<UNK>'
        self.PREDICTED = 'predicted'
        self.CORRECT = 'correct'
        self.k = k

    '''
    Trains a supervised hidden Markov model on a training set.
    self.initial[POS] = log(P(the initial tag is POS))
    self.transition[POS1][POS2] =
    log(P(the current tag is POS2|the previous tag is POS1))
    self.emission[POS][word] =
    log(P(the current word is word|the current tag is POS))
    '''
    def train(self, train_set):
        # Construct index dicts
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # Split the document into sentences
                    lines = f.readlines()
                for line in lines:
                    tagged_tokens = line.split()
                    for pair in tagged_tokens:
                        # Split the pair by the last occurence of '/'
                        token, tag = pair.rsplit('/', 1)
                        token = token.lower()
                        tag = tag.lower()
                        # Add indices for the current token and tag, if has not been added before
                        if not token in self.word_dict:
                            self.word_dict[token] = len(self.word_dict)
                        if not tag in self.pos_dict:
                            self.pos_dict[tag] = len(self.pos_dict)
                            self.reverse_pos_dict[len(self.reverse_pos_dict)] = tag
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
                        token = token.lower()
                        tag = tag.lower()
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
        self.initial += self.k
        self.emission += self.k
        self.transition += self.k
        # Compute probabilities 
        self.initial = self.initial / self.initial.sum()
        self.emission = self.emission / self.emission.sum(axis = 1, keepdims = True)
        self.transition = self.transition / self.transition.sum(axis = 1, keepdims = True)
        # Validate probabilities sum to one
        self.__validate_probabilities()
        # Convert to log probabilities
        self.initial = np.log(self.initial)
        self.emission = np.log(self.emission)
        self.transition = np.log(self.transition)

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        # Convert tokens in the sentence into indices
        # print(sentence)
        sentence = sentence.lower()
        token_indices = [self.word_dict.get(token, self.word_dict[self.UNK]) for token in sentence.split()]
        if len(token_indices) == 0:
            return
        # Initialize v and backpointer
        num_of_states = len(self.pos_dict)
        time_steps = len(token_indices)
        v = np.zeros([num_of_states, time_steps])
        backpointer = np.zeros([num_of_states, time_steps], dtype = np.int32) - 1
        # initialization step
        first_token_index = token_indices[0]
        first_col = np.add(self.initial, self.emission[:, first_token_index])
        v[:, 0] = first_col
        # recursion step
        for i in range(1, len(token_indices)):
            prev_time_step = v[:, i - 1].reshape(v.shape[0], 1)
            precursor = np.add(prev_time_step, self.transition)
            max_prob = np.max(precursor, axis = 0)
            arg_max = np.argmax(precursor, axis = 0)
            token_index = token_indices[i]
            v[:, i] = np.add(max_prob, self.emission[:, token_index])
            backpointer[:, i] = arg_max
        # print(v)
        # print(backpointer)
        # termination step
        best_path = []
        prev_tag = np.argmax(v[:, -1])
        best_path.insert(0, prev_tag)
        for i in reversed(range(1, time_steps)):
            prev_tag = backpointer[prev_tag, i]
            best_path.insert(0, prev_tag)
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
        next_sent_index = 0
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    lines = f.readlines()
                for line in lines:
                    # Parse correct_tags, as a list
                    tagged_tokens = line.split()
                    if not tagged_tokens:
                        continue
                    tagged_tokens = [token.rsplit('/', 1) for token in tagged_tokens]
                    correct_tags = [pair[1] for pair in tagged_tokens]
                    # Predict tags using token
                    sentence = " ".join([pair[0] for pair in tagged_tokens])
                    predicted_tags_indices = self.viterbi(sentence)
                    predicted_tags = [self.reverse_pos_dict[i] for i in predicted_tags_indices]
                    # Add correct and predicted tags to results
                    cur_result = defaultdict(list)
                    cur_result[self.CORRECT] = correct_tags
                    cur_result[self.PREDICTED] = predicted_tags
                    results[next_sent_index] = cur_result
                    next_sent_index += 1
        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        num_total_words = 0.0
        num_correct_predict = 0.0
        for _, result in results.items():
            correct = result[self.CORRECT]
            predicted = result[self.PREDICTED]
            for i in range(len(correct)):
                if correct[i] == predicted[i]:
                    num_correct_predict += 1
                num_total_words += 1
        accuracy = num_correct_predict / num_total_words
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
    for k in range(1, 100):
        pos = POSTagger(k)
        # make sure these point to the right directories
        pos.train('train')
        results = pos.test('dev')
        # pos.train('train_small')
        # results = pos.test('test_small')
        with open('record.csv', 'a') as f:
            accuracy = pos.evaluate(results)
            f.write(str(k) + ',' + str(accuracy) + '\n')
        print('Accuracy:', pos.evaluate(results))
    # pos = POSTagger(1)
    # # make sure these point to the right directories
    # pos.train('train')
    # results = pos.test('dev')
    # # pos.train('train_small')
    # # results = pos.test('test_small')
    # print('Accuracy:', pos.evaluate(results))
