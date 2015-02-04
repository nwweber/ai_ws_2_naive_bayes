# author: Niklas Weber

# Load the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os

# reading the email data from the directories
def read_directory(data_path):
    email_data_list = []
    for fname in os.listdir(data_path):
        fpath = os.path.join(data_path, fname)
        with open(fpath, encoding="utf8") as f:
            try:
                contents = f.read()
                email_data_list.append(contents)
            except UnicodeDecodeError:
                # this is a dirty hack
                # but figuring out encodings is a dirty business
                pass
    return email_data_list


enron1_ham_path = os.path.join("data", "enron1", "ham")
enron1_spam_path = os.path.join("data", "enron1", "spam")

enron1_ham_list = read_directory(enron1_ham_path)
enron1_spam_list = read_directory(enron1_spam_path)

# creating the dictionary

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

enron1_ham_feature_frame = convert_to_feature_space(enron1_ham_list, word_dict)
enron1_spam_feature_frame = convert_to_feature_space(enron1_spam_list, word_dict)


def calc_accuracy(predicted_series, real_series):
    assert len(predicted_series) == len(real_series)
    correct_series = predicted_series * real_series
    correct = correct_series.sum()
    total = len(predicted_series)
    return correct/total


class NBClassifier():

    def __init__(self, word_dict):
        self.phi_y = None
        self.phi_x_y_1 = None
        self.phi_x_y_0 = None
        self.word_dict = word_dict

    def fit(self, data_frame, label_series):
        # compute phi_y
        spam_count = label_series.sum()
        ham_count = len(label_series) - spam_count
        self.phi_y = spam_count / len(label_series)

        # compute phi_x_y_1
        self.phi_x_y_1 = {}
        words_index = data_frame.columns
        data_frame["the label"] = label_series
        for word in words_index:


    def predict(self, data_frame):
        raise NotImplementedError


nb_classifier = NBClassifier(word_dict)
enron2_real = None
enron1_combined_feature_frame = None
enron1_combined_labels = None
enron2_combined_feature_frame = None
nb_classifier.fit(enron1_combined_feature_frame, enron1_combined_labels)
enron2_pred = nb_classifier.predict(enron2_combined_feature_frame)

acc = calc_accuracy(enron2_pred, enron2_real)

print("the accuracy is", acc)