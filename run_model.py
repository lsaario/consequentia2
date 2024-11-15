#!pip install tpot
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re

import train_test

train_ids = train_test.relevant_ids_train + train_test.irrelevant_ids_train
test_ids = train_test.relevant_ids_test + train_test.irrelevant_ids_test
train_test_ids = train_ids + test_ids

train = train_test.build_train()
test = train_test.build_test()

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
        if counter >= 100:
            break

df0 = pd.DataFrame({
    "id" : ids,
    "text" : texts,
    "relevance" : relevances
})

df = pd.concat([train, test, df0], axis=0, ignore_index=True)

vectorizer = TfidfVectorizer(max_features=10000)
td_idfs = vectorizer.fit_transform(df["text"])

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=100)
topics = lda.fit_transform(td_idfs)

topics_df = pd.DataFrame(topics, columns=lda.get_feature_names_out())
#features = vectorizer.get_feature_names_out()

vocab = vectorizer.get_feature_names_out()
for i, comp in enumerate(lda.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

#print(topics_df.shape)
#print(topics_df)

#print(df.head)
#print(td_idfs.head)

df = pd.concat([df, topics_df], axis=1)

#print(df.head)
#print(df.shape)

df1 = df.set_index("id").drop(columns=["text"])

#print(df1.head)
#print(df1.shape)

train = df1.loc[train_test.relevant_ids_train + train_test.irrelevant_ids_train]
X_train = train.drop(columns=["relevance"])
y_train = train["relevance"].astype('int')

test = df1.loc[train_test.relevant_ids_test + train_test.irrelevant_ids_test]
X_test = test.drop(columns=["relevance"])
y_test = test["relevance"].astype('int')

#print(X_train.shape)
#print(y_train.shape)
#print(y_train)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
X_rest = df1.drop(train_test_ids).drop(columns=["relevance"])
print(classifier.score(X_test, y_test))

predictions = classifier.predict(X_rest)
print(predictions)
print(X_rest[predictions == 1])