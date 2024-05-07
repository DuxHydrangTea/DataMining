from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.naive_bayes import GaussianNB



# OUTLOOK: 0-sunny, 1-overcast, 2-rain
# TEMP: 0-hot, 1-cool, 2-mild
# HUMI: 0-high, 1-normal
# WIND: 0-weak, 1-strong 

t1 = [0,0,0,0]
t2 = [0,0,0,1]
t3 = [1,0,0,0]
t4 = [2,2,0,0]
t5 = [2,1,1,0]
t6 = [2,1,1,1]
t7 = [1,1,1,1]
t8 = [0,2,0,0]
t9 = [0,1,1,0]
t10 = [2,2,1,0]
t11 = [0,2,1,1]
t12 = [1,2,0,1]
t13 = [1,0,1,0]
t14 = [2,2,0,1]

train_data = np.array([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14])
label = np.array(['N','N','Y','Y','Y','N','Y','N','Y','Y','Y','Y','Y','N'])
# du lieu kiem tra
d5 = np.array([[2,0,0,0]])
#d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
## Su dung mo hinh theo phan pho xac suat
clf = GaussianNB()
#clf = MultinomialNB()
# Thu hien huan luyen
clf.fit(train_data, label)
# thuc hien test
print('Du doan lop cua d5:', str(clf.predict(d5)[0]))
#print('Xac suat cua d6 tron moi lop:', clf.predict_proba(d6))