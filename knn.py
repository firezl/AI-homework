from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np


mnist_data = fetch_openml("mnist_784")

labels = np.array(list(map(lambda x: int(x), mnist_data['target'])))
images = np.array(mnist_data['data'])/255.0

x_train, x_test,  y_train, y_test = train_test_split(
    images, labels, test_size=0.2)

K = 2

classifier = KNeighborsClassifier(n_neighbors=K)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))
print(np.sum(y_test != y_pred))
