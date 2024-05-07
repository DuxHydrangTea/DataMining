#Đoạn 1
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes1.csv", header=None, names=col_names)
#print(pima.head())

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree','skin']
X = pima[feature_cols]
y = pima.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# clf_tree = DecisionTreeClassifier()
# clf_tree = clf_tree.fit(X_train,y_train)
# y_pred_tree = clf_tree.predict(X_test)
# #print('Accuracy Decision Tree:',metrics.accuracy_score(y_test, y_pred_tree))






def rDClass(criterion = "gini", depth = None):
    rf = RandomForestClassifier(criterion= criterion, max_depth= depth)
    rf = rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)
    print('Accuracy rForest:',round(metrics.accuracy_score(y_test, y_pred_rf),3))

def rDClass_mDepth(criterion = 'gini',m_depth = None):
    for i in  range(m_depth):
        rDClass(criterion=criterion, depth= m_depth)

print("Random Forest Gini: ")
rDClass_mDepth('gini',10)

print("Random Forest Entropy: ")
rDClass_mDepth('entropy',10)

print("Random Forest Log_loss: ")
rDClass_mDepth('log_loss',10)

