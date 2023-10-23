import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.impute import SimpleImputer

##########################
#  Feature computation
##########################

class FeatureComputer():
    def __init__(self,documents):
        self.docs = self.load_documents(documents)
        self.vocab = self.extract_vocabulary()
        self.idf = self.compute_idf(self.docs)
        self.vocab_index = self.get_vocab_index()

    def simple_features(self,document):
        """ Compute the simple features, i.e., number of sentences,
        the average number of words per sentence,
        and the average number of characters per word. """
        sentences = sent_tokenize(document)
        num_sent = len(sentences)
        mean_words = np.mean([len(word_tokenize(sent)) for sent in sentences])
        mean_chars = np.mean([len(word) for word in word_tokenize(document)])
        return num_sent, mean_words, mean_chars

    def load_documents(self,documents):
        """ Index and load documents """
        results = {}
        index = 0
        for doc, label in documents:
            results[index] = {'words':Counter(word_tokenize(doc)), 'label':label, 'doc':doc}
            index += 1
        return results

    """ Compute a dictionary indexing the vocabulary """
    # def extract_vocabulary(self):
    #     vocab = {}
    #     for key,val in self.docs.items():
    #         for word in val['words'].keys():
    #             vocab[word].add(key)
    #     return vocab

    def extract_vocabulary(self):
        vocab = Counter()
        for key, val in self.docs.items():
            vocab.update(val['words'])
        return vocab

    def get_vocab_index(self):
        return {word: idx for idx, word in enumerate(self.vocab)}


    # def get_vocab_index(self): # TODO
    #     """ Build vocabulary index dict """
    #     iterator = 0
    #     result = {}
    #     return result

    # def compute_idf(self, documents): # TODO
    #     """ Compute inverse document frequency dict for all words across
    #     all documents"""
    #     results = {}
    #     for word,keys in self.vocab.items():
    #         results[word] = 0
    #     return results

    def compute_idf(self, documents):
        """ Compute inverse document frequency dict for all words across
        all documents"""
        results = {}
        num_docs = len(documents)
        for word, freq in self.vocab.items():
            results[word] = np.log(num_docs / (1 + freq))
        return results


    def get_features_train(self):
        """ Compute training features for training data """
        examples = {}
        for doc, document in sorted(self.docs.items()):
            feature = np.zeros(len(self.vocab_index))
            feature = np.append(feature, self.simple_features(document['doc']))
            for word, count in document['words'].items():
                if word in self.vocab_index:
                    feature[self.vocab_index[word]] = self.idf[word] * count
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples

    # def get_features_test(self, testdata): # TODO
    #     examples = {}
    #     for doc, document in sorted(testdata):
    #         feature = np.zeros(len(self.vocab_index))
    #         feature = np.append(feature, self.simple_features(document['doc']))
    #         for word, count in document['words'].items():
    #             feature[self.vocab_index[word]] = np.nan # Words which are not existent in the test data at all, but present in the training data (and thus have an entry) still require a real-number value. Thus, we approximate their value using the simple imputer. For this, we set the values to numpy.nan .
    #         examples[doc] = {'feature':feature,
    #                          'label':document['label']}
    #     return examples

    def get_features_test(self, testdata):
        examples = {}
        test_docs = self.load_documents(testdata)
        for doc, document in sorted(test_docs.items()):
            feature = np.zeros(len(self.vocab_index))
            feature = np.append(feature, self.simple_features(document['doc']))
            for word, count in document['words'].items():
                if word in self.vocab_index:
                    feature[self.vocab_index[word]] = self.idf[word] * count
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples

##########################
# Simple helper functions
##########################


def get_number_of_words(text : list):
    return sum([1 for _ in text])

def get_number_of_characters(text):
    iterator = 0
    character_counter = 0
    while iterator < len(text):
        for word in text[iterator]:
            for character in word:
                character_counter = character_counter + 1
            iterator = iterator + 1
    return character_counter

# def read_data(data):
#     result = []
#     log = open(data,'rw')
#     lines = log.readlines()
#     firstline = True # Skip firstline, since it contains the description of the text columns
#     for l in lines:
#         if firstline:
#             continue
#         result.append((line.split('\t')[0],line.split('\t')[1]))
#     return data

def read_data(data):
    result = []
    with open(data, 'r') as log:
        lines = log.readlines()
        for line in lines[1:]:  # Skip the first line, since it contains the description of the text columns
            parts = line.strip().split('\t')
            result.append((parts[0], parts[1]))
    return result


def get_best_features(data): # TODO
    """ Computes the best feature """
    features = np.array([0 for document,_ in data]).reshape(-1,1)
    labels = [y for _,y in data]
    return features, labels


##########################
#       Classifier
##########################

path = os.getcwd()

print ("Loading data...")
train = read_data('train.tsv')
test = read_data('test.tsv')

print ("Computing features...")

feature_comp = FeatureComputer(train)
data_train = feature_comp.get_features_train()
data_test = feature_comp.get_features_test(test)

# Imputer for missing values in the test data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit([doc['feature'] for key,doc in data_train.items()])

train_X = [doc['feature'] for key,doc in sorted(data_train.items())]
train_y = [doc['label'] for key, doc in sorted(data_train.items())]

test_X = imputer.transform([doc['feature'] for key,doc in sorted(data_test.items())])
test_y = [doc['label'] for key, doc in sorted(data_test.items())]

logistic_model = LogisticRegression()
mlp_model = MLPClassifier()

print ("Training models...")

# TODO: train models


# TODO: compute score of two models on the test data

best_train_X, train_y = get_best_features(train)
best_test_X, test_y = get_best_features(test)

logistic_model = LogisticRegression()


# best_model_logistic = LogisticRegression()
mlp_model = MLPClassifier()

logistic_model.fit(best_train_X, train_y)
mlp_model.fit(best_train_X,train_y)

# TODO: predictions


