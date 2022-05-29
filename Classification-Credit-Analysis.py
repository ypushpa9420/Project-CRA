# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:00:30 2022

@author: Pushpa Yadav
"""

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# sns
import seaborn as sns
# util
import utils
import pickle


##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('./credit_risk_dataset.csv')
print("Done ...")


##############################################################
# Exploratory Data Analytics
##############################################################

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

#First, let’s go ahead and check for missing values in our dataset.

print(df.isnull().sum())


#We can see that employment length and interest rate both have missing values. 
#Given that the missing values represent a small percentage of the dataset, we will remove the rows that contain missing values.


#Dropping missing values
df = df.dropna(axis=0)


# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
#Loan status — binary, numerical variable; 0 (no default) or 1 (default) → this is going to be our target variable
clsVars = "loan_status"
print("\n*** Class Variable ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Class names
print("\n*** Unique Class - Categoric Numeric ***")
lnLabels = df[clsVars].unique()
print(lnLabels)


##############################################################
# Data Transformation
##############################################################


# transformations
# change as required
# convert alpha categoric to numeric categoric
# change as required
print("\n*** Transformations ***")
# leperson_home_ownership = preprocessing.LabelEncoder()
# print(df['person_home_ownership'].unique())
# print(df.groupby(df['person_home_ownership']).size())
# df['person_home_ownership'] = leperson_home_ownership.fit_transform(df['person_home_ownership'])
# print(df['person_home_ownership'].unique())
# print(df.groupby(df['person_home_ownership']).size())



print(df['person_home_ownership'].unique())
print(df.groupby(df['person_home_ownership']).size())
df['person_home_ownership']=df['person_home_ownership'].map({'MORTGAGE':0,'OTHER':1,'OWN':2,'RENT':3})
print(df['person_home_ownership'].unique())
print(df.groupby(df['person_home_ownership']).size())

# MORTGAGE (0)
# OTHER (1)
# OWN (2)
# RENT (3)



# leloan_intent = preprocessing.LabelEncoder()
# print(df['loan_intent'].unique())
# print(df.groupby(df['loan_intent']).size())
# df['loan_intent'] = leloan_intent.fit_transform(df['loan_intent'])
# print(df['loan_intent'].unique())
# print(df.groupby(df['loan_intent']).size())

print(df['loan_intent'].unique())
print(df.groupby(df['loan_intent']).size())
df['loan_intent']=df['loan_intent'].map({'DEBTCONSOLIDATION':0,'EDUCATION':1,'HOMEIMPROVEMENT':2,'MEDICAL':3,'PERSONAL':4,'VENTURE':5})
print(df['loan_intent'].unique())
print(df.groupby(df['loan_intent']).size())

# DEBTCONSOLIDATION (0)
# EDUCATION (1)
# HOMEIMPROVEMENT (2)
# MEDICAL (3)
# PERSONAL (4)
# VENTURE (5)

# leloan_grade = preprocessing.LabelEncoder()
# print(df['loan_grade'].unique())
# print(df.groupby(df['loan_grade']).size())
# df['loan_grade'] = leloan_grade.fit_transform(df['loan_grade'])
# print(df['loan_grade'].unique())
# print(df.groupby(df['loan_grade']).size())

print(df['loan_grade'].unique())
print(df.groupby(df['loan_grade']).size())
df['loan_grade']=df['loan_grade'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6})
print(df['loan_grade'].unique())
print(df.groupby(df['loan_grade']).size())

# A (0)
# B (1)
# C (2)
# D (3)
# E (4)
# F (5)
# G (6)


# lecb_person_default_on_file = preprocessing.LabelEncoder()
# print(df['cb_person_default_on_file'].unique())
# print(df.groupby(df['cb_person_default_on_file']).size())
# df['cb_person_default_on_file'] = lecb_person_default_on_file.fit_transform(df['cb_person_default_on_file'])
# print(df['cb_person_default_on_file'].unique())
# print(df.groupby(df['cb_person_default_on_file']).size())

# N (0)
# Y (1)

print(df['cb_person_default_on_file'].unique())
print(df.groupby(df['cb_person_default_on_file']).size())
df['cb_person_default_on_file']=df['cb_person_default_on_file'].map({'N':0,'Y':1})
print(df['cb_person_default_on_file'].unique())
print(df.groupby(df['cb_person_default_on_file']).size())

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# #Removing outliers
df = df[df["person_age"]<=100]
df = df[df["person_emp_length"]<=100]
df = df[df["person_income"]<= 4000000]

# # handle outlier
# print('\n*** Handle Outliers ***')
# colNames = ['person_age','person_emp_length','person_income']
# for colName in colNames:
#       colType =  df[colName].dtype  
#       df[colName] = utils.HandleOutliers(df[colName])
#       if df[colName].isnull().sum() > 0:
#           df[colName] = df[colName].astype(np.float64)
#       else:
#           df[colName] = df[colName].astype(colType)    
# print("Done ...")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if require

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())


# # handle nulls
# colNames = ['RI','Na','Mg','Al','Si','Ca']
# for colName in colNames:
#     colType =  df[colName].dtype  
#     df[colName] = df[colName].fillna(df[colName].mean())
#     df[colName] = df[colName].astype(colType)    
# print("Done ...")

# # check nulls
# print('\n*** Columns With Nulls ***')
# print(df.isnull().sum()) 

# # feature selection
# print("\n*** Feature Scores - XTC ***")
# print(utils.getFeatureScoresXTC(df, clsVars))

# print("\n*** Feature Scores - SKC ***")
# print(utils.getFeatureScoresSKB(df, clsVars))

# # drop cols
# # change as required
# print("\n*** Drop Cols ***")
# df = df.drop('cb_person_cred_hist_length', axis=1)
# print("Done ...")


# # # normalize data
# # print('\n*** Normalize Data ***')
# df = utils.NormalizeData(df, clsVars)
# # print('Done ...')

# # check variance
# print('\n*** Variance In Columns ***')
# print(df.var())

# # check std dev 
# print('\n*** StdDev In Columns ***')
# print(df.std())

# # check nulls
# print('\n*** Columns With Nulls ***')
# print(df.isnull().sum()) 


# Save dataframe to pickled pandas object
df.to_pickle("./finaldataframe.pkl")


##############################################################
# Visual Data Anlytics
##############################################################

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
# plot histograms
print("\n*** Histogram Plot ***")
colNames = df.columns.tolist()
colNames.remove(clsVars)
print('Histograms')
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()
    
    
# all categoic variables except clsVars
# change as required
colNames = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()
    
# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()


################################
# Classification 
# set X & y
###############################

# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# over sampling
print("\n*** Over Sampling Process ***")
X, y = utils.getOverSamplerData(X, y)
print("Done ...")

# counts
print("\n*** Counts ***")
unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

################################
# Classification 
# Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.33, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))

################################
# Classification 
# actual model ... create ... fit ... predict
###############################

# original
# import all model & metrics
print("\n*** Importing Models ***")
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
print("Done ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
lModels = []
lModels.append(('SVM-Clf', SVC(random_state=707)))
lModels.append(('RndFrst', RandomForestClassifier(random_state=707)))
lModels.append(('KNN-Clf', KNeighborsClassifier()))
lModels.append(('LogRegr', LogisticRegression(random_state=707)))
lModels.append(('DecTree', DecisionTreeClassifier(random_state=707)))
lModels.append(('GNBayes', GaussianNB()))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Classification 
# Cross Validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=707)
    # actual corss validation
    cvAccuracy = cross_val_score(oModelObj, X, y, cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(vModelName)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
maxXVIndx = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",maxXVIndx)
print("Model Name : ",xvModNames[maxXVIndx])
print("XVAccuracy : ",xvAccuracy[maxXVIndx])
print("XVStdDev   : ",xvSDScores[maxXVIndx])
print("Model      : ",lModels[maxXVIndx])
 

################################
# Classification 
# evaluate : Accuracy & Confusion Metrics
###############################

# print original confusion matrix
print("\n*** Confusion Matrix ***")
cm = confusion_matrix(y_test, y_test)
print("Original")
print(cm)

# blank list to hold info
print("\n*** Confusion Matrix - Init ***")
cmModelInf = []
cmModNames = []
cmAccuracy = []
print("\nDone ... ")

# iterate through the modes and calculate accuracy & confusion matrix for each
print("\n*** Confusion Matrix - Compare ***")
for vModelName, oModelObj in lModels:
    # blank model object
    model = oModelObj
    # fit the model with train dataset
    model.fit(X_train, y_train)
    # predicting the Test set results
    p_test = model.predict(X_test)
    # accuracy
    vAccuracy = accuracy_score(y_test, p_test)
    # confusion matrix
    cm = confusion_matrix(y_test, p_test)
    # X-axis Predicted | Y-axis Actual
    print("")
    print(vModelName)
    print(cm)
    print("Accuracy", vAccuracy*100)
    # update lists for future use 
    cmModelInf.append((vModelName, oModelObj, cmAccuracy))
    cmModNames.append(vModelName)
    cmAccuracy.append(vAccuracy)

# conf matrix summary
print("\n*** Confusion Matrix Summary ***")
# header
msg = "%7s: %10s " % ("Model", "xvAccuracy")
print(msg)
# for each model
for i in range(0,len(cmModNames)):
    # print accuracy mean & std
    msg = "%8s: %5.7f" % (cmModNames[i], cmAccuracy[i]*100)
    print(msg)

print("\n*** Best CM Accuracy Model ***")
maxCMIndx = cmAccuracy.index(max(cmAccuracy))
print("Index      : ",maxCMIndx)
print("Model Name : ",cmModNames[maxCMIndx])
print("CMAccuracy : ",cmAccuracy[maxCMIndx]*100)
print("Model      : ",lModels[maxCMIndx])


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", lModels[maxXVIndx]) 
print("Confusion Matrix")
print("Accuracy:", cmAccuracy[maxCMIndx])
print("Model   :", lModels[maxCMIndx]) 

# classifier object
# select best cm acc ... why
print("\n*** Classfier Object ***")
model = lModels[maxCMIndx][1]
print(model)
# fit the model
model.fit(X_train,y_train)
print("Done ...")

# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
p_test = model.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)
  
################################
# Final Prediction
# Create Knn Object from whole data
# Read .prd file
# Predict Species
# Confusion matrix with data in .prd file
###############################

# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data-prd.csv')
print(dfp.info())


print(dfp['person_home_ownership'].unique())
print(df.groupby(dfp['person_home_ownership']).size())
dfp['person_home_ownership']=dfp['person_home_ownership'].map({'MORTGAGE':0,'OTHER':1,'OWN':2,'RENT':3})
print(dfp['person_home_ownership'].unique())
print(df.groupby(dfp['person_home_ownership']).size())

# MORTGAGE (0)
# OTHER (1)
# OWN (2)
# RENT (3)


print(dfp['loan_intent'].unique())
print(df.groupby(dfp['loan_intent']).size())
dfp['loan_intent']=dfp['loan_intent'].map({'DEBTCONSOLIDATION':0,'EDUCATION':1,'HOMEIMPROVEMENT':2,'MEDICAL':3,'PERSONAL':4,'VENTURE':5})
print(dfp['loan_intent'].unique())
print(df.groupby(dfp['loan_intent']).size())

# DEBTCONSOLIDATION (0)
# EDUCATION (1)
# HOMEIMPROVEMENT (2)
# MEDICAL (3)
# PERSONAL (4)
# VENTURE (5)


print(dfp['loan_grade'].unique())
print(df.groupby(dfp['loan_grade']).size())
dfp['loan_grade']=dfp['loan_grade'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6})
print(dfp['loan_grade'].unique())
print(df.groupby(dfp['loan_grade']).size())

# A (0)
# B (1)
# C (2)
# D (3)
# E (4)
# F (5)
# G (6)



print(dfp['cb_person_default_on_file'].unique())
print(df.groupby(dfp['cb_person_default_on_file']).size())
dfp['cb_person_default_on_file']=dfp['cb_person_default_on_file'].map({'N':0,'Y':1})
print(dfp['cb_person_default_on_file'].unique())
print(df.groupby(dfp['cb_person_default_on_file']).size())

# N (0)
# Y (1)

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(dfp))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(dfp))

# #Removing outliers
dfp = dfp[dfp["person_age"]<=100]
dfp = dfp[dfp["person_emp_length"]<=100]
dfp = dfp[dfp["person_income"]<= 4000000]


# # handle outlier
# print('\n*** Handle Outliers ***')
# colNames = ['person_age','person_emp_length','person_income']
# for colName in colNames:
#       colType =  dfp[colName].dtype  
#       dfp[colName] = utils.HandleOutliers(dfp[colName])
#       if dfp[colName].isnull().sum() > 0:
#           dfp[colName] = dfp[colName].astype(np.float64)
#       else:
#           dfp[colName] = dfp[colName].astype(colType)    
# print("Done ...")

# check zeros
print('\n*** Columns With Zeros ***')
print((dfp==0).sum())

# check variance
print('\n*** Variance In Columns ***')
print(dfp.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(dfp.std())

# check mean
print('\n*** Mean In Columns ***')
print(dfp.mean())

# check nulls
print('\n*** Columns With Nulls ***')
print(dfp.isnull().sum()) 


# split into data & outcome
print("\n*** Data For Prediction - X & y Split ***")
print(allCols)
print(clsVars)
X_pred = dfp[allCols].values
y_pred = dfp[clsVars].values
print(X_pred)
print(y_pred)

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[maxXVIndx])
print("Model   :", lModels[maxXVIndx]) 
print("Confusion Matrix")
print("Accuracy:", cmAccuracy[maxCMIndx])
print("Model   :", lModels[maxCMIndx]) 

# classifier object
# select best cm acc ... why
print("\n*** Classfier Object ***")
model = lModels[maxCMIndx][1]
print(model)
# fit the model
model.fit(X,y)
print("Done ...")

# predict from model
print("\n*** Actual Prediction ***")
p_pred = model.predict(X_pred)
# actual
print("Actual")
print(y_pred)
# predicted
print("Predicted")
print(p_pred)

# accuracy
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_pred, p_pred)*100
print(accuracy)

# confusion matrix - actual
cm = confusion_matrix(y_pred, y_pred)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix - predicted
cm = confusion_matrix(y_pred, p_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_pred, p_pred)
print(cr)

# update data frame
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
print("Done ...")


################################
# save model & vars as pickle icts
###############################

# classifier object
# select best cm acc ... why
print("\n*** Classfier Object ***")
model = lModels[maxCMIndx][1]
print(model)
# fit the model
model.fit(X,y)
print("Done ...")

# save model
print("\n*** Save Model ***")
filename = './model.pkl'
pickle.dump(model, open(filename, 'wb'))
print("Done ...")

# save vars as dict
print("\n*** Create Vars Dict ***")
dVars = {}
dVars['clsvars'] = clsVars
dVars['allCols'] = allCols
print(dVars)

# save dvars
print("\n*** Save DVars ***")
filename = './model-dvars.pkl'
pickle.dump(dVars, open(filename, 'wb'))
print("Done ...")
