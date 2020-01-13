import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target


from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8,random_state=0)

train_index, test_index = next(ss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)


clf.score(X_test, y_test)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_test, y_pred)
clf.decision_function(X_test[12:15])

clf.predict(X_test[12:15])

for th in range(-3,7):
    print((clf.decision_function(X_test[12:15]) > th).astype(int))


from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, precision_recall_fscore_support

test_score = clf.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, test_score)
import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
print("AUC= ", auc(fpr, tpr))

plt.plot([0,1],[0,1], linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.01])
plt.ylabel('True positive rate (recall)')
plt.xlabel('False Positive Rate (1-specificity)')

test_score = clf.decision_function(X_test)
precision, recall, _ = precision_recall_curve(y_test, test_score)
plt.plot(recall, precision)

plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.01])
plt.ylabel('Precision')
plt.xlabel('Recall')
