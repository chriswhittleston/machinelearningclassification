import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing

from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import knnplots

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

print "Good to go, all packages installed ok, ready to code."


fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen) # read in the CSV file
dataList = list(csvData) # convert to a list

# print dataList[0:2] # print first two rows of data

dataArray = numpy.array(dataList) # convert to numpy array

# print dataArray

X = dataArray[:,2:32].astype(float) # select features as data from full dataArray and convert to floats

y = dataArray[:,1] # set labels to second column (B/M) - the diagnosis

print "X dimensions: ", X.shape
print "y dimensions: ", y.shape

yFreq = scipy.stats.itemfreq(y)
print yFreq

# plot bar chart of frequencies
#plt.bar(left = 0, height = int(yFreq[0][1]), color = 'red')
#plt.bar(left = 1, height = int(yFreq[1][1]))
#plt.show()

# convert the M/B labels to 1 and 0 for later use
le = preprocessing.LabelEncoder()
#Fit numerical values to the categorical data.
le.fit(y)
#Transform the labels into numerical correspondents.
yTransformed = le.transform(y)
# print the original and transformed arrays to check
print y
print yTransformed

# construct a correlation matrix between columns of X (rowvar=0)
correlationMatrix = numpy.corrcoef(X, rowvar=0)

# plot a heat map of the correlations
#fig, ax = plt.subplots()
#heatmap = ax.pcolor(correlationMatrix, cmap = plt.cm.RdBu)
#plt.show()

# plot a scatter plot of the first two columns (features) - tumor radius and perimeter
# colour (c) according to y labels (B and M)
#plt.scatter(x = X[:,0], y = X[:,1], c=y)
#plt.show()

# plot matrix of scatter plots for features - histograms for self/self
def scatter_plot(X,y):
    # set the figure size (scales by the size of the X input array and a constant)
    plt.figure(figsize = (3*X.shape[1],3*X.shape[1]))
    # loop over every pair of columns
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            # create a subplot for the current pair
            plt.subplot(X.shape[1],X.shape[1],i+1+j*X.shape[1])
            # if self/self (i==j) plot two histograms according to y classification (B or M)
            # and colour accordingly
            if i == j:
                plt.hist(X[:, i][y=="M"], alpha = 0.4, color = 'm',

                  bins = numpy.linspace(min(X[:, i]),max(X[:, i]),30))
                plt.hist(X[:, i][y=="B"], alpha = 0.4, color = 'b',
                  bins = numpy.linspace(min(X[:, i]),max(X[:, i]),30))
                plt.xlabel(i)
            # otherwise plot a scatter plot as before to look at correlation
            else:
                plt.gca().scatter(X[:, i], X[:, j],c=y, alpha = 0.4)
                plt.xlabel(i)
                plt.ylabel(j)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()
# pass in a limited slice of the X array - all rows and just first 5 columns (features)
# y is the classification array (B/M) - why not use yTransformed?

# scatter_plot(X[:,:5],y)

# KNN algorithm

# TESTING
nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)
#nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm="kd_tree").fit(X)
distances, indices = nbrs.kneighbors(X)
print indices[:5]
print distances[:5]

# Applying to breast cancer data
# using the transformed label array (0 = B, 1 = M)

# Predict/classify using the 3 nearest neighbors
# (weighting is uniform/default)
knnK3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knnK3 = knnK3.fit(X, yTransformed)
predictedK3 = knnK3.predict(X)

# Predict/classify using the 15 nearest neighbors
# (weighting is uniform/default)
knnK15 = neighbors.KNeighborsClassifier(n_neighbors=15)
knnK15 = knnK15.fit(X, yTransformed)
predictedK15 = knnK15.predict(X)

# Check to see how many points are classified differently using different n_neighbors
nonAgreement = len(predictedK3[predictedK3 != predictedK15])
print "Number of discrepancies (3 vs 15 nearest neighbours): ", nonAgreement

# weight using the 'distance' method - ascribes more importance to neighbours that are very close
# Predict/classify using the 3 nearest neighbors
# (weighting is distance)
knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
knnWD = knnWD.fit(X, yTransformed)
predictedWD = knnWD.predict(X)

# Check to see how many points are classified differently using the uniform weighting above
nonAgreement = len(predictedK3[predictedK3 != predictedWD])
print "Number of discrepancies (uniform vs distance weights): ", nonAgreement

# Now need to split data into training and test sets so we can judge the accuracy of the predictions
# By default, 25% of the data goes in the test set
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

print "XTrain dimensions: ", XTrain.shape
print "yTrain dimensions: ", yTrain.shape
print "XTest dimensions: ", XTest.shape
print "yTest dimensions: ", yTest.shape

# Apply the nearest neighbours classifier to the training set (XTrain) to train a predictor
# yTrain contains the correct labels
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(XTrain,yTrain)
# Apply this fit to predict the classification of the test set (XTest)
predicted = knn.predict(XTest)

# Construct the confusion matrix by comparing the prediction to the correct labels in yTest
# this is tp, fn
#         fp, fn
mat = metrics.confusion_matrix(yTest, predicted)
print mat

# Print some validation metrics
print metrics.classification_report(yTest, predicted)
print "accuracy: ", metrics.accuracy_score(yTest, predicted)

# Plot the accuracy score for a range (1-310?) of n_neighbors using both uniform and distance weights
# knnplots has been given to us - look at how it works!
#knnplots.plotaccuracy(XTrain, yTrain, XTest, yTest, 310)

# Plot the decision boundaries for different models
#knnplots.decisionplot(XTrain, yTrain, n_neighbors=3, weights="uniform")
#knnplots.decisionplot(XTrain, yTrain, n_neighbors=15, weights="uniform")
#knnplots.decisionplot(XTrain, yTrain, n_neighbors=3, weights="distance")
#knnplots.decisionplot(XTrain, yTrain, n_neighbors=15, weights="distance")

# Cross validate result using k-fold method
# This splits the training data in XTrain 5 times, the validation scores are then given as the mean

# n_neighbors = 3
knn3scores = cross_validation.cross_val_score(knnK3, XTrain, yTrain, cv = 5)
print knn3scores
print "Mean of scores KNN3", knn3scores.mean()
print "SD of scores KNN3", knn3scores.std()

# n_neighbors = 15
knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv = 5)
print knn15scores
print "Mean of scores KNN15", knn15scores.mean()
print "SD of scores KNN15", knn15scores.std()

# looking how accuracy means and standard deviations vary with number of folds, k
#  lists to hold results
meansKNNK3 = []
sdsKNNK3 = []
meansKNNK15 = []
sdsKNNK15 = []

# consider 2 to 20 folds
ks = range(2,21)

# calculate validation scores for each ks
for k in ks:
    knn3scores = cross_validation.cross_val_score(knnK3, XTrain, yTrain, cv = k)
    knn15scores = cross_validation.cross_val_score(knnK15, XTrain, yTrain, cv = k)

    # append results to lists
    meansKNNK3.append(knn3scores.mean())
    sdsKNNK3.append(knn3scores.std())
    meansKNNK15.append(knn3scores.mean())
    sdsKNNK15.append(knn3scores.std())

# plot the mean accuracy for KNN3 and KNN15 as number of folds varies
plt.plot(ks, meansKNNK3, label="KNN3 mean accuracy", color="purple")
plt.plot(ks, meansKNNK15, label="KNN15 mean accuracy", color="yellow")
plt.legend(loc=3)
plt.ylim(0.5, 1)
plt.title("Accuracy means with increasing number of folds K")
plt.show()

# plot the standard deviation...
# plot the results
plt.plot(ks, sdsKNNK3, label="KNN3 sd accuracy", color="purple")
plt.plot(ks, sdsKNNK15, label="KNN15 sd accuracy", color="yellow")
plt.legend(loc=3)
plt.ylim(0, 0.1)
plt.title("Accuracy standard deviations with increasing number of folds K")
plt.show()

# Tuning parameters (number of neighbors and weight method) using a grid search
parameters = [{'n_neighbors':[1,3,5,10,50,100],
               'weights':['uniform','distance']}]
clf = GridSearchCV(neighbors.KNeighborsClassifier(), parameters, cv=10, scoring="f1")
clf.fit(XTrain, yTrain)

# print best parameter set
print "Best parameter set found: "
print clf.best_estimator_
