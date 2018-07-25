import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#====================================================================================================#
#Reading the dataset from the dataset text file and put it into an numpy array.
Dataset = []
with open('EEG_Dataset.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        Dataset.append(inner_list)
Dataset = np.array(Dataset)
#Dividing the dataset to features and label
X = Dataset[:,:14]
Y = Dataset[:,14]
#Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Standarize the features and scaling them
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#Training with SVM
svm = SVC(kernel='rbf', random_state=0, C=1) #SVM modeling
svm.fit(X_train, y_train)
#Training with Random Forests
forest = RandomForestClassifier(n_estimators=1000,random_state=0)#, n_jobs=-1)
forest.fit(X_train, y_train)
# Training with Decision Tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=40, random_state=0)
tree.fit(X_train, y_train)

#==================================================================================#

# Finiding the accuracy of each one of the above classifiers
# --------#
# For random forests
training_accuracy_forest= forest.score(X_train,y_train)
test_accuracy_forest = forest.score(X_test,y_test)
#======#
# For decision tree
training_accuracy_tree= tree.score(X_train,y_train)
test_accuracy_tree = tree.score(X_test,y_test)
#======#
# For dsvm
training_accuracy_svm= svm.score(X_train,y_train)
test_accuracy_svm = svm.score(X_test,y_test)
#======#
# Feature Labels
feat_labels = np.array(['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'])
# Feature Importances
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
#Printing them
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
#Plotting them in a bar chart
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# Gather all the accuracies of above classifiers in one list and print it
Accuracy = ['test_accuracy_forest',test_accuracy_forest,'training_accuracy_forest',training_accuracy_forest,'test_accuracy_tree',test_accuracy_tree,'training_accuracy_tree',training_accuracy_tree, 'test_accuracy_svm',test_accuracy_svm,'training_accuracy_svm',training_accuracy_svm]
print Accuracy
print "You were easy!"