import pandas as pd
import os
import numpy as np
from numpy.random import shuffle
from copy import deepcopy
from nltk import wordpunct_tokenize
from nltk.stem import PorterStemmer
from sklearn.utils import shuffle
import re


class UDCDataset(object):

    def __init__(self, train_path, val_path, test_path, vocab_path, max_seq_len):
        print('loading dataset...')
        self.tokenizer = Tokenizer(vocab_path, max_seq_len)
        self.test_x, self.test_y = self.__load_data_type(data_path=test_path)
        self.val_x, self.val_y = self.__load_data_type(data_path=val_path)
        self.train_x, self.train_y = self.__load_data_type(data_path=train_path)
        print('loaded data...')

    @property
    def nb_tng(self):
        return len(self.train_x)

    @property
    def nb_val(self):
        return len(self.val_x)

    @property
    def nb_test(self):
        return len(self.test_x)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def __load_data_type(self, data_path):
        data = pd.read_csv(data_path)
        x = data['Context'].values
        x = self.tokenizer.texts_to_sequences(x)
        y = data['Utterance'].values
        y = self.tokenizer.texts_to_sequences(y)
        return x, y

    def train_generator(self, batch_size, max_epochs=None):
        return self.__data_generator(len(self.train_x), self.train_x, self.train_y, batch_size, max_epochs)

    def val_generator(self, batch_size, max_epochs=None):
        return self.__data_generator(len(self.val_x), self.val_x, self.val_y, batch_size, max_epochs)

    def test_generator(self, batch_size, max_epochs=None):
        return self.__data_generator(len(self.test_x), self.test_x, self.test_y, batch_size, max_epochs)

    def __data_generator(self, num_datapoints, x, y, batch_size, max_epochs=None):
        """
        Read the h5 buckets

        :param data_type:
        :param batch_size:
        :param max_epochs:
        :return:
        """
        # figure out which dataset to use
        epoch = 0
        while True:
            for batch_num in range(0, num_datapoints, batch_size):
                # calc pointer to next batch
                i = batch_num
                i_end = i + batch_size

                # make sure we always have a batch of at least batch size
                # fill the rest of the batch with zeros
                batch_x = x[i: i_end].astype(np.int32)
                batch_y = y[i: i_end].astype(np.int32)

                # serve only batches of the proper size
                if len(batch_x) == batch_size:
                    yield batch_x, batch_y

            # stop generator once we go over max epochs
            epoch += 1
            if max_epochs is not None and epoch >= max_epochs:
                break


class Tokenizer(object):
    not_found_token = 'NF'

    def __init__(self, vocab_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.word_to_index = {Tokenizer.not_found_token: 0}
        self.index_to_word = {0: Tokenizer.not_found_token}
        self.__load_vocab(vocab_path)

    def __load_vocab(self, vocab_path):
        with open(vocab_path, encoding='utf-8') as f:
            for line in f:
                # strip away the \n for each line
                line = line[:-1]
                self.add_vocab_word(line)

    def vocab_size(self):
        return len(self.word_to_index)

    def add_vocab_word(self, word):
        idx = len(self.word_to_index)
        self.word_to_index[word] = idx
        self.index_to_word[idx] = word

    def text_to_sequence(self, paragraph):
        results = []
        for w in paragraph.split(' '):
            try:
                idx = self.word_to_index[w]
            except Exception as e:
                idx = self.word_to_index[Tokenizer.not_found_token]
                pass

            results.append(idx)

        # reverse and ensure we cut off or pad
        results = results[::-1][:self.max_seq_len]
        results.extend([0] * (self.max_seq_len - len(results)))
        return np.asarray(results)

    def texts_to_sequences(self, texts):
        results = []
        for text in texts:
            result = self.text_to_sequence(text)
            results.append(result)

        return np.asarray(results)

