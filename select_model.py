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

train = build_df(relevant_ids_train + irrelevant_ids_train, relevant_ids_train)
test = build_df(relevant_ids_test + irrelevant_ids_test, relevant_ids_test)

#print(train)
#print(test)

train_test = pd.concat([train, test])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_test["text"])

X_train = X[0:8]
y_train = train["relevance"]

X_test = X[8:]
y_test = test["relevance"]

print(X_train.shape)
print(y_train.shape)

tpot = TPOTClassifier(generations=5, population_size=20, cv=2, verbosity=2, random_state=42, config_dict = 'TPOT sparse')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')