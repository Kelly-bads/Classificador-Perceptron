from sklearn import metrics, datasets
from sklearn.linear_model import Perceptron

X, y = datasets.load_digits(return_X_y=True)
clf = Perceptron(alpha=3.14, max_iter=2000, penalty=None)
clf.fit(X, y)
clf.score(X, y)

predict_ = clf.predict(X)
print('\n Base:')
print(predict_)

array = metrics.confusion_matrix(y, predict_)
print('\n Array:')
for item in array:
    print(item)

precisao = clf.score(X, y)
print('\n Precisão: ', precisao * 100, '%')
