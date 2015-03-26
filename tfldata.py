import csv
import numpy
import scipy
import matplotlib.pyplot as plt
# import seaborn
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
fileName = "Nov09JnyExport_filter.csv"
fileOpen = open(fileName, "rU")
# Import the CSV data
csvData = csv.reader(fileOpen) # read in the CSV file
# Convert the CSV data into a list
dataList = list(csvData) # convert to a list
# Convert the list into a numpy array (of strings!)
dataArray = numpy.array(dataList) # convert to numpy array

# DEBUGGING
# print the first 10 rows (journeys) but only the 16th (start from 0) column - the ticket type
# print dataArray[:11, 15]

# 1) Setting up labels (categories/targets)
# We want to see if we can predict the ticket type used, this is column 16, FinalProduct
labels = dataArray[:, 15]
# We need to convert the label strings (ticket types) to numbers so we can use them later
le = preprocessing.LabelEncoder()
# Fit numerical values to the labels
le.fit(labels)
# Transform the labels into numerical correspondents
labelsTransformed = le.transform(labels)
# print the shape of the original and transformed arrays to check
print labels.shape
print labelsTransformed.shape

# 2) Constructing a useful data array
# Not all elements of the TfL data are useful, we want to construct a data array to hold useful quantities
# These are dayno (1), StartStn (4), EndStation (5), EntTime (6) and ExTime (8).
# First, we create an empty array to load the data into with an extra time to contain the journey duration
# which is simply (EntTime - ExTime)
X = numpy.zeros((len(dataArray), 6), dtype=int)

# dayno
X[:, 0] = dataArray[:, 0].astype(int)
# DEBUGGING
# print X[:, 0]

# StartStn
# As the station names are strings, we need to map them to integers just as for the labels
StartStn = dataArray[:, 3]
le.fit(StartStn)
StartStnTransformed = le.transform(StartStn)
# Now we can insert this transformed array into X
X[:, 1] = StartStnTransformed.astype(int)
# DEBUGGING
#for entry in range(0,50):
#    print StartStn[entry], X[entry, 1]

# Q: make a mapping between stations and an integer using unique?
StationNames = numpy.unique(StartStn)
# make StationMap a dictionary?
StationMap =
# could select like this
print StartStn[StartStn == 'Westminster']


# EndStation
# The same is true for EndStation, but we'd like to use the same mapping as for StartStn

# Insert into X
#X[:, 2] = EndStationTransformed.astype(int)
# DEBUGGING
#for entry in range(0,50):
#    print EndStation[entry], X[entry, 2]

# EntTime

# ExTime

# Duration

#X_EndStation = dataArray[:,4] # select the EndStation column
#X_EndStationFreq = scipy.stats.itemfreq(X_EndStation)
#print X_EndStationFreq
# Check length of this array
#print len(X_EndStationFreq)

# plot a bar chart showing these frequencies (hard to get right - too many bars!)
#plt.figure(figsize=(3, 10))
#ind = numpy.arange(len(X_EndStationFreq))    # the y locations for the groups
#plt.barh(ind, X_EndStationFreq[:, 1].astype(int), color='r')
#plt.title('EndStation')
#ticks = plt.yticks(ind + .5, X_EndStationFreq[:, 0])
#xt = plt.xticks()[0]
#plt.xticks(xt, [' '] * len(xt))
#plt.show()
