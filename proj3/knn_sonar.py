import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
val1=time.time()
filename='sonar.all-data.csv'
dataset=pd.read_csv(filename)
array=dataset.values
X=array[:,0:60].astype(float)
y=array[:,60]
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
KNN_1 = KNeighborsClassifier(n_neighbors=1)
KNN_1.fit(X,y)
Actual = y
Predicted = KNN_1.predict(X)
print(metrics.accuracy_score(Actual, Predicted)*100)
KNN_4 = KNeighborsClassifier(n_neighbors=4)
KNN_4.fit(X,y)
Actual = y
Predicted = KNN_4.predict(X)
print(metrics.accuracy_score(Actual, Predicted)*100)
import matplotlib.pyplot as plt
Range = list(range(1,30))
l1 = []
for i in range(1,30):
    KNN = KNeighborsClassifier(n_neighbors=i)
    Accuracy = cross_val_score(KNN, X, y, cv=10, scoring='accuracy').mean()*100
    l1.append(Accuracy)
plt.plot(Range, l1)
val2=time.time()
print('Elapsed time', val2-val1)
#####
