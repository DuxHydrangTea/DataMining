from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


#doc du lieu tu file csv co tieu de cac cot
data = pd.read_csv("iris1.csv")
print(data.head(50))



#Lay thong tin mo ta
x_data = data.iloc[:,0:-1]



#Lay nhan lop cua bo du lieu, ‘class_type’ la tieu de cot phan lop trong Bo du lieu iris1.csv
y_data = data['class_type'].values

#chuyen gia trị du lieu ve dang so
le = LabelEncoder()
x_data = x_data.apply(le.fit_transform)



#Chia du lieu huan luyen (X_train, y_train)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=1)




# Ap dung mo hinh phan lop
model = GaussianNB()
model.fit(X_train, y_train)
y_du_bao = model.predict(X_test)



#Xac dinh hieu nang bo phan lop theo cac chi so
print('accuracy = ', accuracy_score(y_test, y_du_bao))
cnf_matrix = confusion_matrix(y_test, y_du_bao)
print('Ma trận nhầm lẫn:')
print(cnf_matrix)
