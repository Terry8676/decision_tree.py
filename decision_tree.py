#-------------------------------------------------------------------------
# AUTHOR: Ryan Hernandez
# FILENAME: decision_tree.py
# SPECIFICATION: reads the "contact_lens.csv" file and outputs an appropriate depth-2 tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
age_map = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spec_map = {"Myope": 1, "Hypermetrope": 2}
astig_map = {"No": 1, "Yes": 2}
tear_map = {"Reduced": 1, "Normal": 2}

for row in db:
    X.append([age_map[row[0]], spec_map[row[1]], astig_map[row[2]], tear_map[row[3]]])

#encode the original categorical training classes into numbers and add to the vector Y.
class_map = {"No": 1, "Yes": 2}
for row in db:
    Y.append(class_map[row[4]])

#fitting the depth-2 decision tree to the data using entropy as your impurity measure
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf = clf.fit(X, Y)

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()