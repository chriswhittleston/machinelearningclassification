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

# Open TFL journey data
fileName = "tfleg.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen) # read in the CSV file
dataList = list(csvData) # convert to a list

# print dataList[0:2] # print first two rows of data

dataArray = numpy.array(dataList) # convert to numpy array

print dataArray