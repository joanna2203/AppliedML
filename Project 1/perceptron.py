import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


# extract data from the given location
df = pd.read_csv('https://query.data.world/s/izevyzij3mefhaubaph7p5reunebh2', header=None)
# df = pd.read_csv('training.txt',header = None)

# printing the data sheet
print(df.head())

# plt plot the datasheet/dataset
# fig is variable used to plot the dataset
fig = plt.figure()

# giving 3d projection
ax = fig.add_subplot(111, projection='3d')
y = df.iloc[0:100, 4].values
# assigning values ie., if value is between -1 to 1 it is Iris-setosa
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 1, 2]].values

ax.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
ax.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
ax.scatter(X[:100, 1], X[:100, 2], color='yellow', marker='^', label='target')
ax.set_xlabel('petal lenght')
ax.set_ylabel('sepal lenght')
ax.set_zlabel('target')

# ploting the above data
plt.show()


class perceptron2(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of epochs, i.e., passes over the training dataset.

    Attributes
    ------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of misclassifications in every epoch.
    random_state : int
        The seed of the pseudo random number generator.
    """

    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0.0
            for xi, target in zip(X, y):
                update = self.eta * (self.predict(xi) - target)
                self.w_[1:] -= update * xi
                self.w_[0] -= update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print("joanna accuracy")
            self.accuracy=(1-(errors/100))*100
            print(self.accuracy)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


df = pd.read_csv('https://query.data.world/s/izevyzij3mefhaubaph7p5reunebh2', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# standardize
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# learning rate = 0.01
aln = perceptron2(0.01, 10)
aln.fit(X_std, y)

# decision region plot
plot_decision_regions(X_std, y, classifier=aln)

plt.title('Perceptron')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

ppn = perceptron2(eta=0.1, n_iter=10)
ppn.fit(X, y)

ax = plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs', fontsize='xx-large')
plt.ylabel('Number of misclassifications', fontsize='xx-large')


plt.show()

