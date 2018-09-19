import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# Generate three random clusters of IRIS Dataset using SGD
numb_samples = 30
numb_dims = 2
numb_clusters = 3


labels = ['setosa','virginica','versicolor']

for label in labels:
    df = pd.read_csv('https://query.data.world/s/izevyzij3mefhaubaph7p5reunebh2', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-'+label, -1, 1)
    X = df.iloc[0:100, [0,2]].values
    f = open('iris_'+label+'.csv','w+')
    for i in range(len(X)):
        f.write(str(X[i][0])+","+str(X[i][1])+","+str(y[i])+"\n")
        plt.show()

center_cluster_A = [1,1]
center_cluster_B = [7,7]
center_cluster_C = [5,1]


df = pd.read_csv('https://query.data.world/s/izevyzij3mefhaubaph7p5reunebh2', header=None)
df1=pd.read_csv('')
#define the three clusters
A = 0.6*np.random.randn(numb_samples, numb_dims) + center_cluster_A
B = 0.6*np.random.randn(numb_samples, numb_dims) + center_cluster_B
C = 0.6*np.random.randn(numb_samples, numb_dims) + center_cluster_C

y = df.iloc[0:100, 4].values
Y = np.where(y == 'Iris-versicolor',-1,1)
Y = np.where(y == 'Iris-setosa', -1, 1)
Y = np.where(y=='Iris-virginica',-1,1)
X = df.iloc[0:100, [0, 2]].values

#now concat them into one list and create another list (Y) that indicates the cluster membership of each element in X using One Vs Rest Strategy
X = np.hstack((np.ones(numb_clusters * numb_samples).reshape(numb_clusters * numb_samples, 1), np.vstack((A, B, C))))
Y = np.vstack(((np.zeros(numb_samples)).reshape(numb_samples, 1), np.ones(numb_samples).reshape(numb_samples, 1), 2 * np.ones(numb_samples).reshape(numb_samples, 1)))

total_samples = numb_clusters * numb_samples

# Run gradient descent
eta = 1
max_iter = 1000
w = np.zeros((3, 3))
grad_thresh = 5
for t in range(0, max_iter):
    grad_t = np.zeros((3, 3))
    for i in range(0, total_samples):
        x_i = X[i, :]
        y_i = Y[i]
        exp_vals = np.exp(w.dot(x_i))
        lik = exp_vals[int(y_i)]/np.sum(exp_vals)
        grad_t[int(y_i), :] += x_i*(1-lik)

    w = w + 1/float(total_samples) * eta * grad_t
    grad_norm = np.linalg.norm(grad_t)

    if grad_norm < grad_thresh:
        print ("Converged in ",t+1,"steps.")
        break

    if t == max_iter-1:
        print ("Warning, did not converge.")


# Begin plotting here
# Define our class colors
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])

# Generate the mesh
x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
h = 0.02 # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_mesh = np.c_[np.ones((xx.size, 1)), xx.ravel(), yy.ravel()]
Z = np.zeros((xx.size, 1))

# Compute the likelihood of each cell in the mesh
for i in range(0, xx.size):
    lik = w.dot(X_mesh[i, :])
    Z[i] = np.argmax(lik)

# Plot it
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.plot(X[0:numb_samples - 1, 1], X[0:numb_samples - 1, 2], 'ro', X[numb_samples:2 * numb_samples - 1, 1], X[numb_samples:2 * numb_samples - 1,
                                                                                                            2], 'bo', X[2 * numb_samples:, 1], X[2 * numb_samples:, 2], 'go')
plt.axis([np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, np.min(X[:, 2])-0.5, np.max(X[:, 2])+0.5])
plt.show()


