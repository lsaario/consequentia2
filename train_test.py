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
irrelevant_ids_train = ["A30010", "A78192", "A14868", "B00372", "A03456", "A55481", "B04151", "A79319"]
irrelevant_ids_test  = ["A62724", "A40069", "A92785", "A25407"]

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

###

train_ids = relevant_ids_train + irrelevant_ids_train
test_ids = relevant_ids_test + irrelevant_ids_test
train_test_ids = train_ids + test_ids

train = build_train()
test = build_test()

ids = []
texts = []
relevances = []

counter = 0
rootdir = Path('data/')
for file_path in rootdir.glob('*.P5.txt'):
    file_name = Path(file_path).name
    file_id = re.sub(".P5.txt$", "", file_name)
    if file_id not in train_test_ids:
        file = open(file_path, "r")
        text = file.read()
        ids.append(file_id)
        texts.append(text)
        relevances.append(None)
        counter += 1
        #if counter >= 1000:
            #break

df0 = pd.DataFrame({
    "id" : ids,
    "text" : texts,
    "relevance" : relevances
})

df = pd.concat([train, test, df0], axis=0, ignore_index=True)

def match_bigram(word1, word2, qualifier):
    consequence = "(con)?seq.*"
    condition1 = re.match(qualifier, word1) and re.match(consequence, word2)
    condition2 = re.match(qualifier, word2) and re.match(consequence, word1)
    return not (condition1 or condition2)

def select_formal(word1, word2):
    return match_bigram(word1, word2, "form.*")

def select_material(word1, word2):
    return match_bigram(word1, word2, "mater.*")

def select_logical(word1, word2):
    return match_bigram(word1, word2, "logic.*")

import nltk
from nltk import BigramCollocationFinder
#from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()

def filter_count(row, fn):
    finder = BigramCollocationFinder.from_words(row["text"].split(), window_size = 5)
    finder.apply_ngram_filter(fn)
    #print(finder.score_ngrams(bigram_measures.raw_freq))
    bigram_counts = finder.ngram_fd.items()
    #print(bigram_counts)
    counter = 0
    for bigram_count in bigram_counts:
        counter += bigram_count[1]
    return counter

def colloc_freq_formal(row):
    return filter_count(row, select_formal)

def colloc_freq_material(row):
    return filter_count(row, select_material)

def colloc_freq_logical(row):
    return filter_count(row, select_logical)

#df["colloq_freq"] = df.apply(get_colloc, axis=1)

#print(df.head)
#print(df["colloq_freq"][0].nbest(bigram_measures.raw_freq, 5))

df["formal_consequence"] = df.apply(colloc_freq_formal, axis=1)
df["material_consequence"] = df.apply(colloc_freq_material, axis=1)
df["logical_consequence"] = df.apply(colloc_freq_logical, axis=1)

df1 = df.set_index("id").drop(columns=["text"])

print(df1.head)
print(df1.shape)

train = df1.loc[relevant_ids_train + irrelevant_ids_train]
X_train = train.drop(columns=["relevance"])
y_train = train["relevance"].astype('int')

test = df1.loc[relevant_ids_test + irrelevant_ids_test]
X_test = test.drop(columns=["relevance"])
y_test = test["relevance"].astype('int')

#print(X_train.shape)
#print(y_train.shape)
#print(y_train)

#print(df1.var() - df1.loc[train_test_ids].var())