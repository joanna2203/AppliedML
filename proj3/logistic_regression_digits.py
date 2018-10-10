import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



#load the dataset


val1=time.time()
digits = load_digits()
X = digits.data  #input
y = digits.target #output
print(digits.data.shape)  #1797 samples * 64 (8*8)pixels

fig = plt.figure()
plt.gray()
ax1 = fig.add_subplot(231)
ax1.imshow(digits.images[0])

ax2 = fig.add_subplot(232)
ax2.imshow(digits.images[1])

ax3 = fig.add_subplot(233)
ax3.imshow(digits.images[2])

plt.tight_layout()
plt.show()
log_reg = LogisticRegression()
#train a model
log_reg.fit(X, y)
#sklearn provides several ways to test a classifier
accuracy_score(y, log_reg.predict(X))
print("joanna")
print(accuracy_score)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.90,random_state=100)
from sklearn.linear_model import LogisticRegression
logic=LogisticRegression()
logic.fit(X_train,y_train)
predicted_value=logic.predict(X_test)
logic.score(X_test,y_test)#94.73%accuracy




from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predicted_value))

#another way
log_reg.score( X, y)
#confusion matrix is a table that can be used to evaluate the performance of a classifier
#each row shows actual values and column values shows predicted values
confusion_matrix(y, log_reg.predict(X))
#we can use predict method to predict the class
print("Predicted : " , log_reg.predict(digits.data[1].reshape(1,-1)))
print("Actual : ", digits.target[1])
#we can also predict the probability of each class
proba = log_reg.predict_proba(digits.data[1].reshape(1,-1)) # second column has the highest probability
print(proba)
def sigmoid(x):
    return 1/(1+np.exp(x))
numbers = np.linspace(-20,20,50) #generate a list of numbers
numbers
#we will pass each number through sigmoid function
results = sigmoid(numbers)
results[:10]
np.argmax(proba) #please note index starts with 0
#10 fold cross-validation

digits = load_digits()


model = LogisticRegression()
scores = cross_val_score(model,digits.data, digits.target, cv=10, scoring='accuracy' )
scores.mean()
digits = load_digits()
skfold = StratifiedKFold(n_splits= 10)
costs = []
for train_index,test_index in skfold.split(digits.data, digits.target):
    X_train, y_train = digits.data[train_index], digits.target[train_index]
    X_test, y_test = digits.data[test_index], digits.target[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    costs.append(model.score(X_test, y_test))
np.mean(costs)

#########################################jo
from sklearn import datasets
import numpy as np

iris = datasets.load_digits()
X = load_digits().data[:, [2, 3]]
y = load_digits().target

print('Class labels:', np.unique(y))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardizing the features:

# In[7]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
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
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))



# ### Logistic regression intuition and conditional probabilities

# In[14]:


import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
# plt.savefig('images/03_02.png', dpi=300)
plt.show()





def cost_1(z):
    return - np.log(sigmoid(z))


def cost_0(z):
    return - np.log(1 - sigmoid(z))


z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/03_04.png', dpi=300)
plt.show()


# In[17]:


class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.
    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# In[18]:


X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('images/03_05.png', dpi=300)
plt.show()
val2=time.time()
print('Elapsed Time (s):', val2-val1)
##################

# ### Training a logistic regression model with scikit-learn

