import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=20,
                           n_features=2,
                           n_classes=3,
                           n_clusters_per_class=1,
                           n_informative=2,
                           n_redundant=0,n_repeated=0,random_state=8)


import matplotlib.pyplot as plt
%matplotlib inline
plt.set_cmap(plt.cm.brg)

plt.scatter(X[:,0],X[:,1], c=y, s=50)

def plotBoundary(X, clf, mesh=True, cmap=plt.get_cmap()):
    x_min = min(X[:,0])
    x_max = max(X[:,1])
    y_min = min(X[:,0])
    y_max = max(X[:,1])

    XX, YY = np.megrid[x_min:x_max:200j, y_min:y_max:200j]

    z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    z = z.reshape(XX.shape)

    if mesh:
        plt.pcolormesh(XX, YY, z, zorder=-10, cmap=cmap)

    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=1)

clf.fit(X,y)

plt.scatter(X[:,0],X[:,1], marker='o', s=50, c=y)
plotBoundary(X, clf)


from sklearn import linear_model
clf = linear_model.LogisticRegression()

from sklearn import svm
clf = svm.SVC(kernel='linear', C=10)
