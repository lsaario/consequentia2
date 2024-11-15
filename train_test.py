#!pip install tpot
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re

relevant_ids_train   = ["A53187", "A59232", "A26578", "A01231"]
relevant_ids_test    = ["A16218", "A45277"]
irrelevant_ids_train = ["A30010", "A78192", "A14868", "B00372"]
irrelevant_ids_test  = ["A62724", "A40069"]

def build_df(ids, relevant_ids):
    texts = []
    relevances = []
    for id in ids:
        file = open("data/" + id + ".P5.txt", "r")
        texts.append(file.read())
        file.close()
        relevances.append(int(id in relevant_ids))
    df = pd.DataFrame({
        "id" : ids,
        "text" : texts,
        "relevance" : relevances
    })
    return df
    #print(df.head)

def build_train():
    return build_df(relevant_ids_train + irrelevant_ids_train, relevant_ids_train)

def build_test():
    return build_df(relevant_ids_test + irrelevant_ids_test, relevant_ids_test)