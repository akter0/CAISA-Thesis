import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from sklearn.impute import SimpleImputer


##########################
# Feature computation
##########################

class FeatureComputer():
    def __init__(self, documents):
        self.docs = self.load_documents(documents)
        self.vocab = self.extract_vocabulary()
        self.idf = self.compute_idf()
        self.vocab_index = self.get_vocab_index()

    def simple_features(self, document):
        sentences = sent_tokenize(document)
        num_sent = len(sentences)
        mean_words = np.mean([len(word_tokenize(sent)) for sent in sentences])
        mean_chars = np.mean([len(word) for word in word_tokenize(document)])
        return num_sent, mean_words, mean_chars

    def load_documents(self, documents):
        results = {}
        for index, (doc, label) in enumerate(documents):
            results[index] = {'words': Counter(word_tokenize(doc)), 'label': label, 'doc': doc}
        return results

    def extract_vocabulary(self):
        vocab = Counter()
        for key, val in self.docs.items():
            vocab.update(val['words'])
        return vocab

    def get_vocab_index(self):
        return {word: idx for idx, word in enumerate(self.vocab)}

    def compute_idf(self):
        num_docs = len(self.docs)
        idf = {}
        for word in self.vocab:
            doc_count = sum(1 for doc in self.docs.values() if word in doc['words'])
            idf[word] = np.log(num_docs / (1 + doc_count))
        return idf

    def get_features_train2(self):
        examples = {}
        for doc, document in sorted(self.docs.items()):
            feature = np.zeros(len(self.vocab_index) + 3)  # +3 for the simple features
            feature[:len(self.vocab_index)] = [document['words'][word] * self.idf[word] for word in self.vocab]
            feature[-3:] = self.simple_features(document['doc'])
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples

    def get_features_train(self): # TODO
        """ Coompute training features for training data """
        examples = {}
        for doc, document in sorted(self.docs.items()):
            feature = np.zeros(len(self.vocab_index))
            feature = np.append(feature, self.simple_features(document['doc']))
            for word, count in document['words'].items():
                feature[self.vocab_index[word]] = 0
            examples[doc] = {'feature':feature, 'label':document['label']}
        return examples

    def get_features_test2(self, testdata):
        examples = {}
        for doc, document in sorted(testdata):
            feature = np.zeros(len(self.vocab_index) + 3)
            for word, count in document['words'].items():
                if word in self.vocab:
                    feature[self.vocab_index[word]] = count * self.idf[word]
            feature[-3:] = self.simple_features(document['doc'])
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples

    def get_features_test(self, testdata):
        examples = {}
        test_docs = self.load_documents(testdata)
        for doc, document in sorted(test_docs.items()):
            feature = np.zeros(len(self.vocab_index))
            feature = np.append(feature, self.simple_features(document['doc']))

            for word, count in document['words'].items():
                try:
                    feature[self.vocab_index[
                        word]] = np.nan  # Words which are not existent in the test data at all, but present in the training data (and thus have an entry) still require a real-number value. Thus, we approximate their value using the simple imputer. For this, we set the values to numpy.nan .
                except Exception as ex:
                    print(repr(ex))
                    print(word)
                    print(doc)
            examples[doc] = {'feature': feature, 'label': document['label']}
        return examples


##########################
# Simple helper functions get_best_features
##########################

def read_data(data):
    result = []
    with open(data, 'r') as log:
        lines = log.readlines()
        for line in lines[1:]:  # Skip the first line
            parts = line.strip().split('\t')
            result.append((parts[0], parts[1]))
    return result


##########################
# Classifier
##########################

print("Loading data...")
train = read_data('train.tsv')
test = read_data('test.tsv')

print("Computing features...")

feature_comp = FeatureComputer(train)
data_train = feature_comp.get_features_train()
data_test = feature_comp.get_features_test(test)

# Imputer for missing values in the test data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit([doc['feature'] for key, doc in data_train.items()])

train_X = [doc['feature'] for key, doc in sorted(data_train.items())]
train_y = [doc['label'] for key, doc in sorted(data_train.items())]

test_X = imputer.transform([doc['feature'] for key, doc in sorted(data_test.items())])
test_y = [doc['label'] for key, doc in sorted(data_test.items())]

logistic_model = LogisticRegression()
mlp_model = MLPClassifier()

print("Training models...")

# Train models
logistic_model.fit(train_X, train_y)
mlp_model.fit(train_X, train_y)

# Make predictions
logistic_predictions = logistic_model.predict(test_X)
mlp_predictions = mlp_model.predict(test_X)

# TODO: Compute and report evaluation metrics for the two models


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Compute evaluation metrics
logistic_accuracy = accuracy_score(test_y, logistic_predictions)
mlp_accuracy = accuracy_score(test_y, mlp_predictions)

logistic_precision = precision_score(test_y, logistic_predictions, average='weighted')
mlp_precision = precision_score(test_y, mlp_predictions, average='weighted')

logistic_recall = recall_score(test_y, logistic_predictions, average='weighted')
mlp_recall = recall_score(test_y, mlp_predictions, average='weighted')

logistic_f1 = f1_score(test_y, logistic_predictions, average='weighted')
mlp_f1 = f1_score(test_y, mlp_predictions, average='weighted')

print("Logistic Regression Accuracy: {:.2f}".format(logistic_accuracy))
print("MLP Accuracy: {:.2f}".format(mlp_accuracy))

print("Logistic Regression Precision: {:.2f}".format(logistic_precision))
print("MLP Precision: {:.2f}".format(mlp_precision))

print("Logistic Regression Recall: {:.2f}".format(logistic_recall))
print("MLP Recall: {:.2f}".format(mlp_recall))

print("Logistic Regression F1 Score: {:.2f}".format(logistic_f1))
print("MLP F1 Score: {:.2f}".format(mlp_f1))

# Feature importance analysis
# You can use logistic_model.coef_ and mlp_model.coefs_ to analyze feature importance for each model.
