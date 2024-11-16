#!pip install tpot
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re

import train_test

import numpy as np
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier()
#from sklearn.svm import LinearSVC
#classifier = LinearSVC()
classifier.fit(train_test.X_train, train_test.y_train)
print(classifier.score(train_test.X_test, train_test.y_test))

#X_rest = train_test.df1.drop(train_test.train_test_ids).drop(columns=["relevance"])
X = train_test.df1.drop(columns=["relevance"])
train_test.df1["prediction"] = classifier.predict(X)

train_test.df1.to_csv("predictions.csv")
#print(predictions)
#print(X_rest[predictions == 1])