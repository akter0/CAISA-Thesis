import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer  # Import the missing SimpleImputer
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

# FeatureComputer class
class FeatureComputer:
    def __init__(self, documents):
        self.docs = self.load_documents(documents)
        self.vocab = self.extract_vocabulary()
        self.idf = self.compute_idf(self.docs)
        self.vocab_index = self.get_vocab_index()

    def simple_features(self, document):
        sentences = sent_tokenize(document)
        num_sent = len(sentences)
        mean_words = np.mean([len(word_tokenize(sent)) for sent in sentences])
        mean_chars = np.mean([len(word) for word in word_tokenize(document)])
        return num_sent, mean_words, mean_chars

    def load_documents(self, documents):
        results = {}
        for doc, label in documents:
            results[doc] = {'words': Counter(word_tokenize(doc)), 'label': label, 'doc': doc}
        return results

    def extract_vocabulary(self):
        vocab = Counter()
        for val in self.docs.values():
            vocab.update(val['words'])
        return vocab

    def get_vocab_index(self):
        return {word: i for i, word in enumerate(self.vocab.keys())}

    def compute_idf(self, documents):
        num_docs = len(documents)
        idf = {}
        for word, freq in self.vocab.items():
            idf[word] = np.log(num_docs / (1 + freq))
        return idf

    def get_features_train(self):
        examples = {}
        for doc, document in self.docs.items():
            feature = np.zeros(len(self.vocab_index))
            for word, count in document['words'].items():
                feature[self.vocab_index[word]] = count * self.idf[word]
            num_sent, mean_words, mean_chars = self.simple_features(document['doc'])
            feature = np.append(feature, [num_sent, mean_words, mean_chars])
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples

    def get_features_test(self, testdata):
        examples = {}
        test_docs = self.load_documents(testdata)
        for doc, document in sorted(test_docs.items()):
            feature = np.zeros(len(self.vocab_index))
            for word, count in document['words'].items():
                if word in self.idf:
                    feature[self.vocab_index[word]] = count * self.idf[word]
                # else:
                #     self.vocab_index[word] = 0
                    # feature[self.vocab_index[word]] = 0  # Handle unseen words
            num_sent, mean_words, mean_chars = self.simple_features(document['doc'])
            feature = np.append(feature, [num_sent, mean_words, mean_chars])
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples

# Helper functions
def read_data(data):
    result = []
    with open(data, 'r') as log:
        lines = log.readlines()
        firstline = True
        for line in lines:
            if firstline:
                firstline = False
                continue
            result.append((line.split('\t')[0], line.split('\t')[1]))
    return result

def get_best_features(data):
    features = np.array([0 for _, _ in data]).reshape(-1, 1)
    labels = [y for _, y in data]
    return features, labels

# Load data
path = os.getcwd()
print("Loading data...")
train = read_data('train.tsv')
test = read_data('test.tsv')

# Compute features
print("Computing features...")
feature_comp = FeatureComputer(train)
data_train = feature_comp.get_features_train()
data_test = feature_comp.get_features_test(test)

# Imputer for missing values in the test data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit([doc['feature'] for doc in data_train.values()])

train_X = [doc['feature'] for doc in data_train.values()]
train_y = [doc['label'] for doc in data_train.values()]

test_X = imputer.transform([doc['feature'] for doc in data_test.values()])
test_y = [doc['label'] for doc in data_test.values()]

# Train models
logistic_model = LogisticRegression()
mlp_model = MLPClassifier()

logistic_model.fit(train_X, train_y)
mlp_model.fit(train_X, train_y)

# Predictions
logistic_predictions = logistic_model.predict(test_X)
mlp_predictions = mlp_model.predict(test_X)
