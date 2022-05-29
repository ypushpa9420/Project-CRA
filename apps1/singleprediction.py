# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:00:42 2021

@author: Pushpa Yadav
"""
# import streamlit
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd

    

def app():
    
    # hides all warnings
    
    global clsVars, allCols, Le 

    ################################
    # model
    ################################
    
    
    # load model
    #print("\n*** Load Model ***")
    import pickle
    filename = './model.pkl'
    with open(filename, 'rb') as file:
      
    # Call load method to deserialze
        myvar = pickle.load(file)
  
    model = myvar
    #print(model)
    #print("Done ...")
    
    # load vars
    #print("\n*** Load Vars ***")
    filename = './model-dvars.pkl'
    with open(filename, 'rb') as file:
      
    # Call load method to deserialze
        myvar1 = pickle.load(file)
  
    dVars = myvar1
    #print(dVars)
    clsVars = dVars['clsvars'] 
    allCols = dVars['allCols']
    #print("Done ...")
    
    
    
    ################################
    # predict
    #######N#######################
    
    def getPredict(dfp):
        
        
        dfp['person_home_ownership']=dfp['person_home_ownership'].map({'MORTGAGE':0,'OTHER':1,'OWN':2,'RENT':3})
        
        # MORTGAGE (0)
        # OTHER (1)
        # OWN (2)
        # RENT (3)
        
        
        dfp['loan_intent']=dfp['loan_intent'].map({'DEBTCONSOLIDATION':0,'EDUCATION':1,'HOMEIMPROVEMENT':2,'MEDICAL':3,'PERSONAL':4,'VENTURE':5})
        
        # DEBTCONSOLIDATION (0)
        # EDUCATION (1)
        # HOMEIMPROVEMENT (2)
        # MEDICAL (3)
        # PERSONAL (4)
        # VENTURE (5)
        
        
        dfp['loan_grade']=dfp['loan_grade'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6})
        
        # A (0)
        # B (1)
        # C (2)
        # D (3)
        # E (4)
        # F (5)
        # G (6)
        
        
        
        dfp['cb_person_default_on_file']=dfp['cb_person_default_on_file'].map({'N':0,'Y':1})
        
        # N (0)
        # Y (1)


        X_pred = dfp[allCols].values
        #y_pred = dfp[clsVars].values
        #print(X_pred)
        #print(y_pred)
    
        # predict from model
        #print("\n*** Actual Prediction ***")
        p_pred = model.predict(X_pred)
        # actual
        #print("Actual")
        #print(y_pred)
        # predicted
        #print("Predicted")
        #print(p_pred)
    
    
        # update data frame
        #print("\n*** Update Predict Data ***")
        dfp['Predict'] = p_pred
        #dfp[clsVars] = leSpc.inverse_transform(dfp[clsVars])
        #print("Done ...")
    
        return (dfp)
    
     
    ########################################################
    # headers
    ########################################################
    
    # title
    st.title("Predict your data here!!!!")
    
    # title
    st.sidebar.title("Credit Risk Assessment")
    
    # user inputs
    person_age = st.sidebar.number_input('Age', min_value=20, max_value=99, format="%i")
    person_income = st.sidebar.number_input('Income', min_value=4000, max_value=4000000, format="%i")
    person_home_ownership = st.sidebar.radio("Home status",('MORTGAGE', 'OTHER', 'OWN', 'RENT'))
    person_emp_length = st.sidebar.number_input('Employment length',  min_value=0, max_value=100,format="%i")
    loan_intent = st.sidebar.radio("Loan intent",('DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL','PERSONAL','VENTURE'))
    loan_grade = st.sidebar.radio("Loan grade",('A', 'B', 'C', 'D','E','F','G'))
    loan_amnt = st.sidebar.number_input('Loan amount', min_value=500, max_value=35000,format="%i")
    loan_int_rate = st.sidebar.number_input('Interest rate' ,min_value=5.42, max_value=23.22 ,format="%f")
    loan_percent_income = st.sidebar.number_input('Loan to income ratio',min_value=0.0, max_value=0.83 ,format="%f")
    cb_person_default_on_file = st.sidebar.radio("Historical default",('N', 'Y'))
    cb_person_cred_hist_length = st.sidebar.number_input('Credit history length', min_value=2, max_value=30, format="%i")

    
    # submit
    if(st.sidebar.button("Submit")):
        # create data dict ... colNames should be kay
        data = {'person_age': person_age,
                'person_income' : person_income,
                'person_home_ownership': person_home_ownership,
                'person_emp_length' : person_emp_length,
                'loan_intent':loan_intent,
                'loan_grade':loan_grade,
                'loan_amnt':loan_amnt,
                'loan_int_rate':loan_int_rate,
                'loan_percent_income':loan_percent_income,
                'cb_person_default_on_file':cb_person_default_on_file,
                'cb_person_cred_hist_length':cb_person_cred_hist_length}
        # create data frame for predict
        dfp = pd.DataFrame(data, index=[0])
        # show dataframee', min_value=20, max_value=99
        st.subheader('Input Data')
        st.write('person_age : ', person_age)
        st.write('person_income  : ', person_income)
        st.write('person_emp_length : ', person_emp_length)
        st.write('loan_intent : ', loan_intent)
        st.write('loan_grade : ', loan_grade)
        st.write('loan_amnt : ', loan_amnt)
        st.write('loan_int_rate : ', loan_int_rate)
        st.write('loan_percent_income : ', loan_percent_income)
        st.write('cb_person_default_on_file : ', cb_person_default_on_file)
        st.write('cb_person_cred_hist_length : ', cb_person_cred_hist_length)
        # predict
        st.dataframe(dfp)
        dfp = getPredict(dfp)
        # show dataframe
        st.subheader('Prediction')
        if dfp['Predict'][0]==0:
            st.write('Customer data is good and loan can be recover based on current data')

        if dfp['Predict'][0]==1:
            st.write('Risk is involved if loan is given to customer')

        # reset    
        
        st.button("Reset")
