import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics  


def printMatrixConfustion(y,db):
    cnf_matrix = metrics.confusion_matrix(y, db)
    print('Ma trận nhầm lẫn:')
    print(cnf_matrix)

# show accuracy
def printDestree(cri = 'gini', split = 'best'):  
    myGini = DecisionTreeClassifier(criterion=cri, splitter=split)
    myGini = myGini.fit(X_train,y_train)
    y_gi_pred = myGini.predict(X_test)
    print(f"Accuracy {cri}:",metrics.accuracy_score(y_test, y_gi_pred))
    printMatrixConfustion(y_test,y_gi_pred)


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin','bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes1.csv", header=None, names=col_names)
#print(pima.head())



feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols]
y = pima.label
#print(pima)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Literal['gini', 'entropy', 'log_loss'] = "gini"

#gini
printDestree('gini', 'random')
printDestree('entropy', 'random')
printDestree('log_loss', 'random')
