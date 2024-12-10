import numpy as np
import pandas as pd
#import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.preprocessing import OneHotEncoder
import math
import  time
import pickle


#Load players data
data = pd.read_csv('player-value-prediction.csv')


#miss data number
missing_columns=['wage','release_clause_euro','club_rating','club_jersey_number','value']
#outlier in data
"""""
for i in missing_columns:
    oulier=find_outliers_IQR(data[i])
    print("number of outliers: "+ i +" "+str(len(oulier)))
    print(oulier)
"""""
mean_column=['club_rating','club_jersey_number']
for i in mean_column:
    data[i] = data[i].fillna(data[i].mean())
medain_column=['wage','release_clause_euro','value']

for i in medain_column:
    data[i] = data[i].fillna(data[i].median())

#split coloumn postion to 4 column each column express about one postaion
data[['pos1', 'pos2', 'pos3','pos4']]=data['positions'].str.split(',', expand=True)
postion=['pos1', 'pos2', 'pos3','pos4','tags','club_team','traits']
for i in postion:
    data[i] = data[i].replace(np.nan, 0)

#convert category to numerical by encoder
fencode=['nationality','club_team','traits']
data=Feature_Encoder(data,fencode)

#convet category to numerical by one hot encoder
categrey_columns=['preferred_foot','work_rate','body_type','club_position','pos1', 'pos2', 'pos3','pos4','tags']
for feature in categrey_columns:
    data = encode_and_bind(data, feature)

#convert date to year in club_join_date
data['club_join_date'] = data['club_join_date'].replace(np.nan, 0)
for element in range(len(data)):
    x=data.at[element, 'club_join_date']
    if x==0:
        continue
    s=x.split("/")
    data.at[element, 'club_join_date']=int(s[2])

#convert date to year in contract_end_year
data['contract_end_year'] = data['contract_end_year'].replace(np.nan, 0)
for element in range(len(data)):
    x = data.at[element, 'contract_end_year']
    if x==0:
        continue
    l=len(x)
    if l==4:
        data.at[element, 'contract_end_year']=int(x)
        continue
    s=x.split("-")
    year=2000+int(s[2])
    data.at[element, 'contract_end_year']=int(year)

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

#fill zero value to median
date_missing_columns=['club_join_date','contract_end_year']
for ele in date_missing_columns:
    data[ele] = data[ele].replace(0, data[ele].median())
for j in pos:
    data[j] = data[j].replace(0, data[j].median())

#Drop coloumn and make value coloumn =-1
value=data['value']
data=data.drop(labels=['id', 'name','full_name','birth_date','national_team',
    'national_rating','national_team_position','national_jersey_number','positions','value'], axis=1)
data=pd.concat([data, value], axis=1)

#Feature
X=data.iloc[:,0:259]
#column value
Y=data['value']
print(X)
print("max value of player"+str(data['value'].max()))
print("min value of player"+str(data['value'].min()))

#print(X)
#print(Y)


#get colleration betwen features
corr = data.corr()
#top 50% colleration traning feature with the value
top_feature = corr.index[abs(corr['value']>0.5)]

#colleration plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
print(top_feature)
X = X[top_feature]

#feature scalling
X = featureScaling(X,0,1)

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True,random_state=10)

poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()

start=time.time()
poly_model.fit(X_train_poly, y_train)
stop=time.time()
print(f"Training time: {stop-start}s")

#save model
#model_filename = "polynomial_regression.sav"
#saved_model = pickle.dump(poly_model, open(model_filename,'wb'))
#print('Model is saved into to disk successfully Using Pickle')

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

#ypred=poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction2 = poly_model.predict(poly_features.fit_transform(X_test))



print('Mean Square Error for training data-set', metrics.mean_squared_error(y_train, y_train_predicted))
print('Mean Square Error for test data-set', metrics.mean_squared_error(y_test, prediction2))
MSE=metrics.mean_squared_error(y_test, prediction2)
print('ROOT Mean Square Error for test data-set', math.sqrt(MSE))

print("accuracy for test = ", round(metrics.r2_score(y_test, prediction2), 2))


true_car_price=np.asarray(y_test)[0]
predicted_car_price=prediction2[0]
print('True value for the first player : ' + str(true_car_price))
print('Predicted value for the first player : ' + str(predicted_car_price))









