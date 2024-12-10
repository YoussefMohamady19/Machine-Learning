import pickle
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score

from Pre_processing2 import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


#feature_seleted=['release_clause_euro', 'RF', 'CB', 'RAM', 'LM', 'age', 'ST', 'club_position_RES', 'pos4_CAM', 'club_position_GK', 'RDM', 'LB', 'nationality', 'pos2_CF', 'club_position_LS', 'pos1_ST', 'heading_accuracy', 'pos1_LW', 'pos3_CAM', 'club_position_CM', 'club_position_RCB', 'club_position_SUB', 'club_position_LM', 'club_position_RDM', 'pos2_LB', 'pos4_CM', 'pos3_RW', 'tags_#Acrobat', 'tags_#Engine', 'pos4_LM', 'club_team', 'work_rate_High/ Low', 'traits', 'pos3_CF', 'pos2_RM', 'pos3_RB', 'pos1_LM', 'tags_0', 'pos2_CM', 'pos1_RB', 'pos1_CAM', 'pos1_LB', 'pos1_CM', 'strength', 'pos4_0', 'pos4_CF', 'body_type_Lean', 'sprint_speed', 'pos3_LW', 'pos3_LB', 'pos3_CM', 'preferred_foot_Right', 'height_cm', 'tags_#Speedster,#Acrobat', 'pos3_ST', 'pos3_LM', 'pos4_ST', 'pos3_LWB', 'tags_#Speedster', 'tags_#Engine,#Strength', 'tags_#Engine,#Acrobat']
#Load players data
data = pd.read_csv('player-tas-classification-test.csv')

X_test,Y_test=preProcessing(data)
fs = SelectKBest(score_func=f_classif, k=61)
X_selected = fs.fit_transform(X_test, Y_test)

model_filename = "Logistic Regression.sav"
multi = pickle.load(open(model_filename, 'rb'))
print("Loaded Succsufuly")
prediction=multi.predict(X_selected)
print(f'Accuracy on Model : {round(accuracy_score(Y_test, prediction)*100, 3)}')

