from sklearn.naive_bayes import MultinomialNB
import numpy as np

'''
    Đau ngực:
        1: có
        0: không
    Hút thuốc:
        1: có
        0: không
    Tập thể dục:
        1: có
        0: không
'''
d1 = [1, 1, 1]
d2 = [0,0,1]
d3 = [0,0,1]
d4 = [1,1,0]
d5 = [1,0,0]

train_data = np.array([d1, d2, d3, d4, d5])
label = np.array([1,1,0,0,0])

dTest = np.array([[1, 1, 1]])
clf = MultinomialNB()
clf.fit(train_data, label)  

print('', str(clf.predict(dTest)[0]))
