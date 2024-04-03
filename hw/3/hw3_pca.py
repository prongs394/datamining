import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_lfw_people


lfw_people = fetch_lfw_people(min_faces_per_person=200 , resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target

n_features = X.shape[1]
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print(target_names)
print("___________________________________________")

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_components = 125
print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))

#print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a SVM classification model
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set
#print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print("SVM:")
print("classfication report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))



"""
# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=5, n_col=8):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

#plt.show()
"""


print("")
print("")
print("MLP:")

#MLP for neural network
def MLP(xtrain, xtest, ytrain, ytest):
    # train a multi-layer perceptron
    # verbose : bool, optional, default False (Whether to print progress messages to stdout)
    # batch_size :  number of samples that will be propagated through the network

    #print("Fitting the classifier to the training set")
    clf = MLPClassifier(hidden_layer_sizes=(400,), batch_size=80, verbose=False, early_stopping=True).fit(xtrain,ytrain)
    y_pred = clf.predict(xtest)
    print("classfication report:")
    print(classification_report(ytest, y_pred, target_names=target_names))
    print("confusion matrix:")
    print(confusion_matrix(ytest, y_pred, labels=range(n_classes)))

MLP(X_train , X_test , y_train , y_test)