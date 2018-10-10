from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import time

# for this project, I chose the digits dataset that comes with Sklearn
# the dataset is conveniently split into features(images) and targets already
val1=time.time()
digits=datasets.load_digits()

# First, we reshape the data of the images into a array that will be accepted by the classifier,
# essentially turning it into a column vector
num = len(digits.images)
data = digits.images.reshape((num, -1))

# Next, we split the data into a training set and testing set using train_test_split
x_train, x_test, y_train, y_test = train_test_split(
data, digits.target, train_size=0.4, test_size=0.6)

# Next we instantiate 2 SVM classifiers, one using a linear kernel, the other using a 'rbf' or radial basis function
# kernel in order to compare the difference
classifier1=svm.SVC(kernel='linear')
classifier1.fit(x_train,y_train)

classifier2=svm.SVC(kernel='rbf')
classifier2.fit(x_train, y_train)

# Next we test the 2 classifiers with the testing set using cross_val_predict and output the accuracy score using
# metrics.accuracy_score
predicted1 = cross_val_predict(classifier1,x_test, y_test)
predicted2 = cross_val_predict(classifier2,x_test, y_test)
print('Result Linear: ', metrics.accuracy_score(y_test, predicted1))
print('Results RBF: ', metrics.accuracy_score(y_test, predicted2))
####################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_digits()
# Take the first two features. We could avoid this by using a two-dim dataset
X = digits.data[:, :2]
y = digits.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
val2=time.time()
print('Elapsed time',val2-val1)
