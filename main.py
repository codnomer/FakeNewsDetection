#Library Definition
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

#Read CSV FİLE
df = pd.read_csv('C:\\Users\\news.csv')
df.shape
headPic = df.head()
print(headPic)

labels = df.label
labelPic = labels.head()
print(labelPic)

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.25, random_state=7)

vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
tf_train = vectorizer.fit_transform(x_train)
tf_test = vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train, y_train)

y_pred = pac.predict(tf_test)

score = accuracy_score(y_test, y_pred)
print(f'Accuracy : {round(score*100,2)} %')

cm =confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cm)
