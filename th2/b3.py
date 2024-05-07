from sklearn.naive_bayes import MultinomialNB,BernoulliNB,CategoricalNB,ComplementNB,GaussianNB
import numpy as np

# Du lieu huan luyen
# sunny 0 -- over 1 -- rain 2
# hot 0 -- mild 1 -- cold 2
# high 0 - nor 1
# wek 0 -- str 1
d1 = [0,0,0,0]
d2 = [0,0,0,1]
d3 = [1,0,0,0]
d4 = [2,1,0,0]
d5 = [2,2,1,0]
d6 = [2,2,1,1]
d7 = [1,2,1,1]
d8 = [0,1,0,0]
d9 = [0,2,1,0]
d10 = [2,1,1,0]

train_data = np.array([d1, d2, d3, d4,d5,d6,d7,d8,d9,d10])
# 1 - no // 2 - yes
label = np.array([1,1,2,2,2,1,2,1,2,2])

# du lieu kiem tra
d11 = np.array([[1,1,1,1]])
d12 = np.array([[1,1,0,1]])
d13 = np.array([[1,0,1,0]])
d14 = np.array([[2,1,0,1]])

dTest = [d11,d12,d13,d14]
## Su dung mo hinh theo phan pho xac suat
#clf = MultinomialNB() # 75%
#clf = BernoulliNB() #50%
#clf = CategoricalNB() # 100%
#clf = ComplementNB() # 25%
clf = GaussianNB() #75%

# Thu hien huan luyen
clf.fit(train_data, label)

# thuc hien test
for i in range(4):
    print('\nDu doan lop :', str(clf.predict(dTest[i])[0]))
    print('Xac suat cua d6 tron moi lop:', clf.predict_proba(dTest[i]))


