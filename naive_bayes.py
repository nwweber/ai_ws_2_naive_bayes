# author: Niklas Weber

# Load the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import pickle

FROM_SCRATCH = True
pickle_path = os.path.join("data", "data.pickle")

print("reading the data")


def read_data():
    # reading the email data from the directories
    def read_directory(data_path):
        email_data_list = []
        for fname in os.listdir(data_path):
            fpath = os.path.join(data_path, fname)
            with open(fpath, encoding="latin1") as f:
                try:
                    contents = f.read()
                    email_data_list.append(contents)
                except UnicodeDecodeError:
                    print("Warning: skipped input file due to encoding errors")
                    # this is a dirty hack
                    # but figuring out encodings is a dirty business
                    pass
        return email_data_list


    enron1_ham_path = os.path.join("data", "enron1", "ham")
    enron1_spam_path = os.path.join("data", "enron1", "spam")

    enron1_ham_list = read_directory(enron1_ham_path)
    enron1_spam_list = read_directory(enron1_spam_path)

    # creating the dictionary
    print("creating the dictionary")
    dict_html_list = read_directory(os.path.join("data", "dictionaries"))
    # set to force uniqueness
    word_dict = set()
    for dict_html in dict_html_list:
        soup = BeautifulSoup(dict_html)
        for li in soup.ul.children:
            word_dict.add(li.text)
    # list to guarantee stable iteration order
    word_dict = list(word_dict)

    # translate email data into dictionary/feature space


    def convert_to_feature_space(email_list, word_dict):
        features = []
        for email in email_list:
            # primitive. words could also be separated by special characters
            email_words = email.split(" ")
            email_features = {word: int(word in email_words) for word in word_dict}
            features.append(email_features)
        # convert to pandas data frame. because why not?
        features_frame = pd.DataFrame(features)
        return features_frame

    print("converting to feature space")
    enron1_ham_feature_frame = convert_to_feature_space(enron1_ham_list, word_dict)
    enron1_spam_feature_frame = convert_to_feature_space(enron1_spam_list, word_dict)

    enron2_ham_path = os.path.join("data", "enron2", "ham")
    enron2_spam_path = os.path.join("data", "enron2", "spam")

    enron2_ham_list = read_directory(enron2_ham_path)
    enron2_spam_list = read_directory(enron2_spam_path)

    enron2_ham_feature_frame = convert_to_feature_space(enron2_ham_list, word_dict)
    enron2_spam_feature_frame = convert_to_feature_space(enron2_spam_list, word_dict)

    enron2_combined_labels = pd.Series(([0] * len(enron2_ham_feature_frame) + ([1] * len(enron2_spam_feature_frame))))
    enron1_combined_labels = pd.Series(([0] * len(enron1_ham_feature_frame) + ([1] * len(enron1_spam_feature_frame))))
    enron1_combined_feature_frame = enron1_ham_feature_frame.append(enron1_spam_feature_frame)
    enron2_combined_feature_frame = enron2_ham_feature_frame.append(enron2_spam_feature_frame)

    out_tuple = enron1_combined_feature_frame, enron1_combined_labels, enron2_combined_feature_frame, enron2_combined_labels

    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(out_tuple, pickle_file)

    return out_tuple


if FROM_SCRATCH:
    print("from scratch")
    enron1_combined_feature_frame, enron1_combined_labels, enron2_combined_feature_frame, enron2_combined_labels = read_data()
else:
    print("from the pickle file", pickle_path)
    with open(pickle_path, "rb") as pickle_file:
        enron1_combined_feature_frame, enron1_combined_labels, enron2_combined_feature_frame, enron2_combined_labels = pickle.load(
            pickle_file)


def calc_accuracy(predicted_series, real_series):
    assert len(predicted_series) == len(real_series)
    correct_series = predicted_series * real_series
    correct = correct_series.sum()
    total = len(predicted_series)
    return correct / total


def calc_bernoulli(x, p):
    return (p ** x) * ((1 - p) ** (1 - x))


class NBClassifier():
    def __init__(self):
        self.phi_y = None
        self.phi_x_y_1 = None
        self.phi_x_y_0 = None

    def fit(self, data_frame, label_series):
        words_index = data_frame.columns
        data_frame["the label"] = label_series

        # compute phi_y
        spam_count = label_series.sum()
        ham_count = len(label_series) - spam_count
        self.phi_y = spam_count / len(label_series)

        # compute phi_x_y_1
        self.phi_x_y_1 = {}
        for word in words_index:
            count = len(data_frame[(data_frame[word] == 1) & (data_frame["the label"] == 1)])
            param = count / spam_count
            self.phi_x_y_1[word] = param

        # compute phi_x_y_0
        self.phi_x_y_0 = {}
        for word in words_index:
            count = len(data_frame[(data_frame[word] == 1) & (data_frame["the label"] == 0)])
            param = count / ham_count
            self.phi_x_y_0[word] = param

    def predict(self, data_frame):
        predictions = []
        words_index = data_frame.columns
        for index, row in data_frame.iterrows():
            p_spam = 0
            p_ham = 0
            temp_ham = 1
            temp_spam = 1
            for word in words_index:
                temp_ham *= calc_bernoulli(row[word], self.phi_x_y_0[word])
                temp_ham *= calc_bernoulli(row[word], self.phi_x_y_1[word])
            p_ham = temp_ham * self.phi_y
            p_spam = temp_spam * self.phi_y
            predictions.append(int(p_spam > p_ham))
        return predictions


nb_classifier = NBClassifier()

print("fitting the model")
nb_classifier.fit(enron1_combined_feature_frame, enron1_combined_labels)

print("predicting")
enron2_pred = nb_classifier.predict(enron2_combined_feature_frame)

acc = calc_accuracy(enron2_pred, enron2_combined_labels)

print("the accuracy is", acc)