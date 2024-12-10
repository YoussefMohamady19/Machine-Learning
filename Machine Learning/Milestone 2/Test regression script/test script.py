import math
import pickle
import pandas as pd
import sklearn
from Pre_processing2 import *

import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

feature_seleted=['overall_rating', 'potential', 'wage', 'international_reputation(1-5)','release_clause_euro', 'club_rating', 'reactions']

#Load players data
data = pd.read_csv('player-tas-regression-test.csv')

X_test,Y_test=preProcessing(data)
X_test=X_test[feature_seleted]

poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_test)

model_filename = "polynomial_regression.sav"
multi = pickle.load(open(model_filename, 'rb'))
print("Loaded Succsufuly")

prediction=multi.predict(X_train_poly)
print('Mean Square Error for test data-set', metrics.mean_squared_error(Y_test, prediction))


