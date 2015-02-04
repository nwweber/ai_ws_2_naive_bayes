# author: Niklas Weber

# Load the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os


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




