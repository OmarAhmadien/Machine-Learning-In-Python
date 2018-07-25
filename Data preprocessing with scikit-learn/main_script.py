import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# section (a) save your list in the variable below
Dataset = []
with open('dataset.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        Dataset.append(inner_list)
Labels=[]
with open('feature_labels.txt') as h:
    for line in h:
        Labels.append([elt.strip() for elt in line.split(',')])
df = pd.DataFrame(Dataset)
df.columns = Labels
df1 = pd.DataFrame(Dataset).replace('?',np.nan)
df1.columns = Labels
l= df1.isnull().sum()
missing_features =l.tolist()
#-------------------------------------------------------------------#
# Converting every column that has numbers to float type.
df1['normalized-losses'] =df1['normalized-losses'].astype(float)
df1['length'] = df1['length'].astype(float)
df1['price'] = df1['price'].astype(float)
df1['stroke'] = df1['stroke'].astype(float)
df1['bore'] = df1['bore'].astype(float)
df1['horsepower']=df1['horsepower'].astype(float)
df1['peak-rpm'] =df1['peak-rpm'].astype(float)
df1['symboling'] = df1['symboling'].astype(float)
df1['highway-mpg'] = df1['highway-mpg'].astype(float)
df1['wheel-base'] = df1['wheel-base'].astype(float)
df1['width'] = df1['width'].astype(float)
df1['height'] = df1['height'].astype(float)
df1['curb-weight'] = df1['curb-weight'].astype(float)
df1['engine-size'] = df1['engine-size'].astype(float)
df1['compression-ratio'] = df1['compression-ratio'].astype(float)
df1['city-mpg'] = df1['city-mpg'].astype(float)
#-------------------------------------------------------------------#
md = df1['num-of-doors'].mode()
df1['num-of-doors'].fillna(md[0],inplace=True)
df1['normalized-losses'].fillna(df1['normalized-losses'].mean(),inplace=True)
df1['price'].fillna(df1['price'].mean(),inplace=True)
df1['bore'].fillna(df1['bore'].mean(),inplace=True)
df1['stroke'].fillna(df1['stroke'].mean(),inplace=True)
df1['peak-rpm'].fillna(df1['peak-rpm'].median(),inplace=True)
df1['horsepower'].fillna(df1['horsepower'].median(),inplace=True)
# section (b) save your dataframe in the variable below
df_b = df1.copy(deep=True)

# section (c) save your dataframe in the variable below
df_c = df_b.copy(deep = True)
door_mapping = {'two': 2,'four': 4}
cylinder_mapping = {'two':2,'three': 3, 'four': 4, 'five' : 5 ,'six': 6 ,'eight':8,'twelve':12}
df_c['num-of-doors'] = df_c['num-of-doors'].replace(door_mapping)
df_c['num-of-cylinders'] = df_c['num-of-cylinders'].replace(cylinder_mapping)
df_c['num-of-cylinders']= df_c['num-of-cylinders'].astype(float)



# section (d) save your dataframe in the variable below
df_d = pd.get_dummies(df_c)

# section (e) save your results in the variables below
X = df_d.drop(['symboling'],axis=1)
y = df_d['symboling']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)


stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=20, random_state=0)
tree.fit(X_train_std, y_train)

training_accuracy= tree.score(X_train_std,y_train)
test_accuracy = tree.score(X_test_std,y_test)
#print training_accuracy,test_accuracy

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=0, max_depth=20)

forest.fit(X_train_std, y_train)
feat_labels = df_d.columns[1:]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
feature_importance = []
for f in range(X_train.shape[1]):
    feature_importance.append((feat_labels[indices[f]],importances[indices[f]]))

#------------ for the most effective feature ----------#
f1_X_train_selected = X_train['num-of-doors'].values
f1_X_train_selected = map(lambda x: [x],f1_X_train_selected)
f1_X_test_selected = X_test['num-of-doors'].values
f1_X_test_selected= map(lambda x: [x],f1_X_test_selected)
tree.fit(f1_X_train_selected, y_train)
f1_training_accuracy_selected= tree.score(f1_X_train_selected,y_train)
f1_test_accuracy_selected = tree.score(f1_X_test_selected,y_test)

#print training_accuracy_selected,test_accuracy_selected
#--------------- for the two most effective features ----#
f12_X_train_selected = X_train[['num-of-doors','length']]
f12_X_test_selected = X_test[['num-of-doors','length']]
tree.fit(f12_X_train_selected, y_train)
f12_training_accuracy_selected= tree.score(f12_X_train_selected,y_train)
f12_test_accuracy_selected = tree.score(f12_X_test_selected,y_test)

#---------------- first three important features ----------#
f123_X_train_selected = X_train[['num-of-doors','length','normalized-losses']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f123_X_test_selected = X_test[['num-of-doors','length','normalized-losses']]
#f12_X_test_selected= map(lambda x: [x],f12_X_test_selected)
tree.fit(f123_X_train_selected, y_train)
f123_training_accuracy_selected= tree.score(f123_X_train_selected,y_train)
f123_test_accuracy_selected = tree.score(f123_X_test_selected,y_test)
#----------------------first four ---------------------------#
f1234_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f1234_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight']]
#f12_X_test_selected= map(lambda x: [x],f12_X_test_selected)
tree.fit(f1234_X_train_selected, y_train)
f1234_training_accuracy_selected= tree.score(f1234_X_train_selected,y_train)
f1234_test_accuracy_selected = tree.score(f1234_X_test_selected,y_test)
#----------------------first 5 ---------------------------#
f12345_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f12345_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price']]
#f12_X_test_selected= map(lambda x: [x],f12_X_test_selected)
tree.fit(f12345_X_train_selected, y_train)
f12345_training_accuracy_selected= tree.score(f12345_X_train_selected,y_train)
f12345_test_accuracy_selected = tree.score(f12345_X_test_selected,y_test)
#----------------------first 6 ---------------------------#
f123456_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f123456_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower']]
#f12_X_test_selected= map(lambda x: [x],f123456_X_test_selected)
tree.fit(f123456_X_train_selected, y_train)
f123456_training_accuracy_selected= tree.score(f123456_X_train_selected,y_train)
f123456_test_accuracy_selected = tree.score(f123456_X_test_selected,y_test)
#----------------------first 7 ---------------------------#
f7_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f7_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height']]
#f12_X_test_selected= map(lambda x: [x],f123456_X_test_selected)
tree.fit(f7_X_train_selected, y_train)
f7_training_accuracy_selected= tree.score(f7_X_train_selected,y_train)
f7_test_accuracy_selected = tree.score(f7_X_test_selected,y_test)
#----------------------first 8 ---------------------------#
f8_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f8_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base']]
#f12_X_test_selected= map(lambda x: [x],f123456_X_test_selected)
tree.fit(f8_X_train_selected, y_train)
f8_training_accuracy_selected= tree.score(f8_X_train_selected,y_train)
f8_test_accuracy_selected = tree.score(f8_X_test_selected,y_test)
#----------------------first 8 ---------------------------#
f9_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base','bore']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f9_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base','bore']]
#f12_X_test_selected= map(lambda x: [x],f123456_X_test_selected)
tree.fit(f9_X_train_selected, y_train)
f9_training_accuracy_selected= tree.score(f9_X_train_selected,y_train)
f9_test_accuracy_selected = tree.score(f9_X_test_selected,y_test)
#----------------------first 8 ---------------------------#
f10_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base','bore','width']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f10_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base','bore','width']]
#f12_X_test_selected= map(lambda x: [x],f123456_X_test_selected)
tree.fit(f10_X_train_selected, y_train)
f10_training_accuracy_selected= tree.score(f10_X_train_selected,y_train)
f10_test_accuracy_selected = tree.score(f10_X_test_selected,y_test)
#----------------------first 8 ---------------------------#
f11_X_train_selected = X_train[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base','bore','width','peak-rpm']]
#f12_X_train_selected = map(lambda x: [x],f12_X_train_selected)
f11_X_test_selected = X_test[['num-of-doors','length','normalized-losses','curb-weight','price','horsepower','height','wheel-base','bore','width','peak-rpm']]
#f12_X_test_selected= map(lambda x: [x],f123456_X_test_selected)
tree.fit(f11_X_train_selected, y_train)
f11_training_accuracy_selected= tree.score(f11_X_train_selected,y_train)
f11_test_accuracy_selected = tree.score(f11_X_test_selected,y_test)

def visualise_line_graph():
    itrain = [f1_training_accuracy_selected,f12_training_accuracy_selected,f123_training_accuracy_selected,f1234_training_accuracy_selected,f1234_training_accuracy_selected,f12345_training_accuracy_selected,f123456_training_accuracy_selected,f7_training_accuracy_selected,f8_training_accuracy_selected,f9_training_accuracy_selected,f10_training_accuracy_selected,f11_training_accuracy_selected]
    htrain = range(1,13)
    plt.plot(htrain, itrain, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.show()
    pass


itest = [f1_test_accuracy_selected, f12_test_accuracy_selected, f123_test_accuracy_selected,
         f1234_test_accuracy_selected, f1234_test_accuracy_selected, f12345_test_accuracy_selected,
         f123456_test_accuracy_selected, f7_test_accuracy_selected, f8_test_accuracy_selected,
         f9_test_accuracy_selected, f10_test_accuracy_selected, f11_test_accuracy_selected]
max = itest.index(max(itest))
# section (g) save your result in the variable below
num_of_features = max + 1


def custom_model_prediction(test_set):
    """This should return the accuracy
    Parameters
    ----------
    test_set: text file representing the test set with
              same format as dataset.txt without symboling column
    Return
    ------
    predictions: type-list """""
    newTest =[]
    with open(test_set) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            newTest.append(inner_list)
    newTest_Labels = []
    with open('feature_newlabels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            newTest_Labels.append(inner_list)

    newdf = pd.DataFrame(newTest)
    newdf.columns = newTest_Labels
    newdf['length'] = newdf['length'].astype(float)
    newdf= newdf.replace('?',np.nan)
    newdf['num-of-doors']= newdf['num-of-doors'].replace(door_mapping)
    newdf['num-of-doors'] = newdf['num-of-doors'].astype(float)
    md = df1['num-of-doors'].mode()
    newdf['num-of-doors'].fillna(md[0], inplace=True)

    newdf['normalized-losses'] = newdf['normalized-losses'].astype(float)
    newdf['normalized-losses'].fillna(newdf['normalized-losses'].mean(), inplace=True)
    if newdf['length'].isnull().sum() >= 1:
        newdf['length'].fillna(newdf['length'].mean(), inplace=True)

    newdf1 = newdf.copy(deep=True)
    T_X_test_selected = newdf1[['num-of-doors', 'length', 'normalized-losses']]
    tree.fit(f123_X_train_selected, y_train)
    pre = tree.predict(T_X_test_selected)
    predictions = np.array(pre).tolist()
    return predictions

"""""
# Ignore this section
all_results = [missing_features, df_b, df_c, df_d, (training_accuracy, test_accuracy),
               feature_importance, num_of_features]
pickle.dump(all_results, open('saved_results.p','wb'))

    
"""""