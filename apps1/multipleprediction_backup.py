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

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

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
        dfp['Predictdata']=dfp['Predict'].map({0:"Not a defaulter",1:'Defaulter'})
        #dfp[clsVars] = leSpc.inverse_transform(dfp[clsVars])
        #print("Done ...")
    
        return (dfp)
    
     
    ########################################################
    # headers
    ########################################################
    
    # title
    st.title("Credit Risk Assessment - Online")
    
    # title
    st.sidebar.title("Credit Risk Assessment")
    
    vFileDets = st.sidebar.file_uploader("Upload Files",type=['csv'])
    if st.sidebar.button("Process"):
        if vFileDets is not None:
            vFileData = {"FileName":vFileDets.name,"FileType":vFileDets.type,"FileSize":vFileDets.size}
            #st.write(vFileData)
            # read csv
            dfp = pd.read_csv(vFileDets)
            # does the file contain clsVars
            try:
                vTemp = dfp[clsVars]
                bClsVars = True
            except:
                bClsVars = False
    		# disp dataframe
            st.subheader('Input Data')
            st.dataframe(dfp)
            # predict
            dfp = getPredict(dfp)
            # show dataframe
            st.subheader('Prediction')
            st.dataframe(dfp)
            csv = convert_df(dfp)
            st.download_button(label="Download Data",data=csv,file_name='dfp.csv',mime='text/csv')
            # show accuracy
            if bClsVars:
                st.subheader('Accuracy')
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(dfp[clsVars], dfp['Predict'])*100
                st.text(accuracy)
            if (st.button('Reset')):
                vFileDets = None
