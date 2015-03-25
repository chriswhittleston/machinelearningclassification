import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#import nbplots

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

# convert the M/B labels to 1 and 0 for later use
le = preprocessing.LabelEncoder()
#Fit numerical values to the categorical data.
le.fit(y)
#Transform the labels into numerical correspondents.
yTransformed = le.transform(y)
# print the original and transformed arrays to check
print y
print yTransformed

# Split X and y into training and test sets
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

print "XTrain dimensions: ", XTrain.shape
print "yTrain dimensions: ", yTrain.shape
print "XTest dimensions: ", XTest.shape
print "yTest dimensions: ", yTest.shape

# Set up a naive bayes model and train it on XTrain using the correct labels in yTrain
nbmodel = GaussianNB().fit(XTrain, yTrain)
# Use it to predict the classification of the data in XTest
predicted = nbmodel.predict(XTest)

# Construct a confusion matrix - yTest is the correct labels to compare against the prediction
mat = metrics.confusion_matrix(yTest, predicted)
print mat

# Print some validation metrics
print metrics.classification_report(yTest, predicted)
print metrics.accuracy_score(yTest, predicted)

#nbplots.decisionplot(XTrain, yTrain)

# Cross validate result using k-fold method
# This splits the training data in XTrain 5 times, the validation scores are then given as the mean
nbscores = cross_validation.cross_val_score(nbmodel, XTrain, yTrain, cv = 5)
print nbscores
print "Mean of scores NB", nbscores.mean()
print "SD of scores NB", nbscores.std()

# looking how accuracy means and standard deviations vary with number of folds, k
#  lists to hold results
meansNB = []
sdsNB = []

# consider 2 to 20 folds
ks = range(2,21)

# calculate validation scores for each ks
for k in ks:
    nbscores = cross_validation.cross_val_score(nbmodel, XTrain, yTrain, cv = k)

    # append results to lists
    meansNB.append(nbscores.mean())
    sdsNB.append(nbscores.std())

# plot the mean accuracy for KNN3 and KNN15 as number of folds varies
plt.plot(ks, meansNB, label="NB mean accuracy", color="purple")
plt.legend(loc=3)
plt.ylim(0.5, 1)
plt.title("Accuracy means with increasing number of folds K")
plt.show()

# plot the standard deviation...
# plot the results
plt.plot(ks, sdsNB, label="NB sd accuracy", color="purple")
plt.legend(loc=3)
plt.ylim(0, 0.1)
plt.title("Accuracy standard deviations with increasing number of folds K")
plt.show()
