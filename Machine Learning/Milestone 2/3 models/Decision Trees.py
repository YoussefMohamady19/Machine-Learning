import matplotlib
import numpy as np
import pandas as pd
#import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from six import StringIO
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
from dython.nominal import associations
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import pickle
import time


# ANOVA feature selection for numeric input and categorical output
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_classif
from featurewiz import featurewiz
from sklearn.preprocessing import StandardScaler


#Load players data
data = pd.read_csv('player-classification.csv')


#miss data number
#'club_team''club_position','club_join_date','contract_end_year','tags', 'traits''LS','ST','RS',
# 'LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB',
missing_columns=['wage','release_clause_euro','club_rating','club_jersey_number']

for i in missing_columns:
    data[i] = data[i].fillna(data[i].mean())


#split coloumn postion
data[['pos1', 'pos2', 'pos3','pos4']]=data['positions'].str.split(',', expand=True)
postion=['pos1', 'pos2', 'pos3','pos4','tags','club_team','traits']
for i in postion:
    data[i] = data[i].replace(np.nan, 0)

#convert
fencode=['nationality','club_team','traits']
data=Feature_Encoder(data,fencode)
#convet category to numerical
categrey_columns=['preferred_foot','work_rate','body_type','club_position','pos1', 'pos2', 'pos3','pos4','tags']
for feature in categrey_columns:
    data = encode_and_bind(data, feature)

#convert date to year
data['club_join_date'] = data['club_join_date'].replace(np.nan, 0)
for element in range(len(data)):
    x=data.at[element, 'club_join_date']
    if x==0:
        continue
    s=x.split("/")
    data.at[element, 'club_join_date']=s[2]

#convert date to year in contract_end_year
data['contract_end_year'] = data['contract_end_year'].replace(np.nan, 0)
for element in range(len(data)):
    x = data.at[element, 'contract_end_year']
    if x==0:
        continue
    l=len(x)
    if l==4:
        continue
    s=x.split("-")
    year=2000+int(s[2])
    data.at[element, 'contract_end_year']=year

#convert categorical data to numerical then summation them
pos=['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']
for ele in pos:
    data[ele] = data[ele].replace(np.nan, 0)
    for element in range(len(data)):
        t = data.at[element, ele]
        if t==0:
            continue
        t2=t.split("+")
        res=int(t2[0])+int(t2[1])
        data.at[element, ele]=res
date_missing_columns=['club_join_date','contract_end_year']
for ele in date_missing_columns:
    data[ele] = data[ele].replace(0, data[ele].median())
for j in pos:
    data[j] = data[j].replace(0, data[j].median())

#Drop coloumn and make value coloumn =-1
value=data['PlayerLevel']
data=data.drop(labels=['id', 'name','full_name','birth_date','national_team',
    'national_rating','national_team_position','national_jersey_number','positions','PlayerLevel'], axis=1)
data=pd.concat([data, value], axis=1)
length=len(data['PlayerLevel'])

X=data.iloc[:,0:259]
Y=data['PlayerLevel']

#feature selection by anova
# # generate dataset
# x, y = make_classification(n_samples=14363, n_features=259, n_informative=5)
# # define feature selection
# fs = SelectKBest(score_func=f_classif, k=5)
# # apply feature selection
# X_selected = fs.fit_transform(X, y)
# print(X_selected.shape)
# print(len(X_selected))

# feature selection
target = 'PlayerLevel'
features, train = featurewiz(data, target, corr_limit=0.5, verbose=2, sep=",",header=0,test_data="", feature_engg="", category_encoders="")
print(features)
print(train)
X_new = train.drop(['PlayerLevel'], axis=1)

# 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, shuffle=True)

#hyper paremeter
params_1 = {'criterion': 'gini', 'splitter': 'best', 'max_depth': 50}
params_2 = {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 70}
params_3 = {'criterion': 'gini', 'splitter': 'random', 'max_depth': 60}
params_4 = {'criterion': 'entropy', 'splitter': 'best', 'max_depth': 80}
params_5 = {'criterion': 'gini', 'splitter': 'best', 'max_depth': 40}
# Separate models
model_1 = DecisionTreeClassifier(**params_1)
model_2 = DecisionTreeClassifier(**params_2)
model_3 = DecisionTreeClassifier(**params_3)
model_4 = DecisionTreeClassifier(**params_4)
model_5 = DecisionTreeClassifier(**params_5)

start=time.time()
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)
model_5.fit(X_train, y_train)
stop=time.time()

print(f"Training time: {stop-start}s")
#save model
model_filename = "Decision_Trees.sav"
saved_model = pickle.dump(model_4, open(model_filename,'wb'))
print('Model is saved into to disk successfully Using Pickle')

start2=time.time()
# Prediction sets
preds_1 = model_1.predict( X_test)
preds_2 = model_3.predict( X_test)
preds_3 = model_3.predict( X_test)
preds_4 = model_4.predict( X_test)
preds_5 = model_5.predict( X_test)
stop2=time.time()

print(f"Testing time: {stop2-start2}s")

print(f'Accuracy on Model 1: {round(accuracy_score(y_test, preds_1), 3)}')
print(f'Accuracy on Model 2: {round(accuracy_score(y_test, preds_2), 3)}')
print(f'Accuracy on Model 3: {round(accuracy_score(y_test, preds_3), 3)}')
print(f'Accuracy on Model 4: {round(accuracy_score(y_test, preds_4), 3)}')
print(f'Accuracy on Model 5: {round(accuracy_score(y_test, preds_5), 3)}')


fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(model_4, fontsize=10)
plt.show()
