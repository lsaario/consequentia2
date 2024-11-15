#!pip install tpot
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import re

import train_test

tpot = TPOTClassifier(generations=5, population_size=20, cv=2, verbosity=2, random_state=42, config_dict = 'TPOT sparse')
tpot.fit(train_test.X_train, train_test.y_train)
print(tpot.score(train_test.X_test, train_test.y_test))
tpot.export('tpot_pipeline.py')