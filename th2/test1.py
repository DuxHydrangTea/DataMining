from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
#doc du lieu tu file csv co tieu de cac cot
#data = pd.read_csv("data\student-por.csv")
#data = pd.read_csv("data\zoo1.csv")
data = pd.read_csv("data\iris1.csv")

print(data.to_string())