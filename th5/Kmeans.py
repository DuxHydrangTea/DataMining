# import numpy as np
# from sklearn.cluster import KMeans
# X = np.array([[5,3],
#  [10,15],
#  [15,12],
#  [24,10],
#  [30,45],
#  [85,70],
#  [71,80],
#  [60,78],
#  [55,52],
#  [80,91],])
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# print(X)
# for item in kmeans.cluster_centers_:
#     print(f"Cluster center: ( {item[0]} , {item[1]} )")
# print(kmeans.labels_)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics




X = 10+ np.random.randn(100, 2) + 8
#print(X)
model_KM = KMeans(n_clusters=4)
model_KM.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=model_KM.labels_,
            cmap='rainbow', label="points")
#plt.show()
inertia = model_KM.inertia_
print(inertia)
s = metrics.silhouette_score(X, model_KM.labels_,
metric='euclidean')
#print(model_KM.labels_)

# for i in range(len(X)):
#     print(f"Sample {i}: ({X[i][0]}, {X[i][1]}) - Label: {model_KM.labels_[i]}")

print(s)