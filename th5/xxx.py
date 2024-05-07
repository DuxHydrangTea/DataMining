

X = [[40,20], [50,50], [60,90], [10,25],[70,70],[60,10],[25,80]]
y = [1,0,0,1,0,1,0]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
neigh.fit(X, y)
print(neigh.predict([[25,30]])[0])