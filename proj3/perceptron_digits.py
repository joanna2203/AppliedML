from sklearn import datasets, metrics
from sklearn.linear_model import Perceptron
import time
# The digits dataset
val1=time.time()
digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

## Q05: You need to adjust this value to increase the score, later.
percentage = 0.8

#set aside the frist 50% of data for the training and the remaining 50% for the testing
X_train = data[:int (n_samples * percentage)]
X_test = data[int (n_samples * (1 - percentage)):]

Y_train = digits.target[:int (n_samples * percentage)]
Y_test =  digits.target[int (n_samples * (1 - percentage)):]

# Create a classifier: a perceptron classifier
classifier = Perceptron(tol=None, max_iter =1000)

# We learn the digits on the first half of the digits
classifier.fit(X_train, Y_train)

# Now predict the value of the digit on the second half:
expected = Y_test
predicted = classifier.predict(X_test)

print("The classification score %.5f\n" % classifier.score(X_test, Y_test))

print("Classification report for classifier %s:\n%s" % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

import matplotlib.pyplot as plt

# Showing the images and prediction from the classifier
images_and_predictions = list(zip(digits.images[int(n_samples * (1 - percentage)):], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:8]):
	plt.subplot(2, 4, index + 1)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Prediction: %i' % prediction)

plt.show()
val2=time.time()
print('Elapsed time ', val2-val1)
