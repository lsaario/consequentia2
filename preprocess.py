from pathlib import Path
import xml.etree.ElementTree as ET

import nltk
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

nltk.download('stopwords')
nltk.download('punkt') # Download the 'punkt' resource
nltk.download('wordnet') # Download the 'wordnet' resource 

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

rootdir = Path('P5_XML_OXF_2015-02/')

def preprocess(dir):
    for file_path in rootdir.glob(dir + '/*.xml'):
        file_name = Path(file_path).name
        print(file_name)

        tree = ET.parse(file_path)
        root = tree.getroot()
        elem = root.find('{http://www.tei-c.org/ns/1.0}text')
        text = ET.tostring(elem, encoding='unicode', method='text')

        # unidecode and convert to lower case
        text = unidecode(text).lower()

        # remove redundant spacing
        text = re.sub(r"[\r\n\t\s]+", " ", text)

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # tokenize
        words = text.split()

        # remove stopwords and numerals
        words = [word for word in words if word not in stop_words and not re.match(r"\d+", word)]

        # lemmatize
        lemmas = [lemmatizer.lemmatize(word) for word in words]

        #print(lemmas)

        # write to file
        f = open("data/" + re.sub(".xml$", ".txt", file_name), "w")
        f.write(" ".join(lemmas))
        f.close()

#preprocess('A0')
#preprocess('A1')
#preprocess('A2')
#preprocess('A3')
#preprocess('A4')
#preprocess('A5')
#preprocess('A6')
#preprocess('A7')
#preprocess('A8')
#preprocess('A9')
#preprocess('B0')
#preprocess('B2')
#preprocess('B3')