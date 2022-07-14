# import streamlit
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        dfp['loan status']=dfp['Predict'].map({0:"Not a defaulter",1:'Defaulter'})
        #dfp[clsVars] = leSpc.inverse_transform(dfp[clsVars])
        #print("Done ...")
    
        return (dfp)
    
     
    ########################################################
    # headers
    ########################################################
    
    # title
    #st.title("Credit Risk Assessment -Bulk future prediction")
    
    # title
    st.sidebar.title("Credit Risk Assessment")

    
    vFileDets = st.sidebar.file_uploader("Upload Files",type=['csv'])
    if vFileDets is None:
            st.header("Thanks for using this tool. Follow below steps to proceed.")
            st.write("1. Please upload file and start your analysis")
            st.write("2. Choose different option to see different cuts of data")
            
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
            #st.subheader('Input Data')
            #st.dataframe(dfp)
            # predict
            dfp = getPredict(dfp)
            # show dataframe
            reportresult = st.sidebar.radio("Choose cuts to see report",('All report data', 'Loan Status', 'Loan Intent','Home Ownership','Loan grade'))


            dfp['person_home_ownership']=dfp['person_home_ownership'].map({0:'MORTGAGE',1:'OTHER',2:'OWN',3:'RENT'})
            dfp['loan_intent']=dfp['loan_intent'].map({0:'DEBTCONSOLIDATION',1:'EDUCATION',2:'HOMEIMPROVEMENT',3:'MEDICAL',4:'PERSONAL',5:'VENTURE'})

            dfp['loan_grade']=dfp['loan_grade'].map({0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'})

            dfp['cb_person_default_on_file']=dfp['cb_person_default_on_file'].map({0:'N',1:'Y'})
          
            
            csv = convert_df(dfp)
            filter1=dfp['loan status']=="Not a defaulter"
            dfp1=dfp.where(filter1)
            filter2=dfp['loan status']=="Defaulter"
            dfp2=dfp.where(filter2)
            pivot = dfp.pivot_table(index =['loan status'],values =['loan_amnt'], aggfunc ='sum')

            
            pivot2 = dfp.pivot_table(index =['loan status','loan_intent'],values =['loan_amnt'], aggfunc ='sum')
            pivot3 = dfp.pivot_table(index =['loan status','person_home_ownership'],values =['loan_amnt'], aggfunc ='sum')
            pivot4 = dfp.pivot_table(index =['loan status','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            pivot5 = dfp.pivot_table(index =['loan status','loan_intent','person_home_ownership'],values =['loan_amnt'], aggfunc ='sum')
            pivot6 = dfp.pivot_table(index =['loan status','loan_intent','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            pivot7 = dfp.pivot_table(index =['loan status','person_home_ownership','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            pivot1 = dfp.pivot_table(index =['loan status','loan_intent','person_home_ownership','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            
            if reportresult=='All report data':
                st.header("Overall data report")
                st.subheader("**Total number of customer applied for loan are as follows**")
                st.dataframe(dfp.groupby('loan status')['loan status'].count())
                st.subheader("Total amount required for loan")
                st.dataframe(pivot)
                st.subheader("**Loan intent for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('loan_intent')['loan status'].count())
                st.subheader("**Loan intent for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('loan_intent')['loan status'].count())
                st.subheader("**Home Ownership for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('person_home_ownership')['loan status'].count())
    
                st.subheader("**Home Ownership for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('person_home_ownership')['loan status'].count())
                st.subheader("**Loan Grade for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('loan_grade')['loan status'].count())
                st.subheader("**Loan Grade for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('loan_grade')['loan status'].count())
                st.subheader("Total amount required for loan based on Loan intent")
                st.dataframe(pivot2)
                st.subheader("Total amount required for loan based on Home Ownership")
                st.dataframe(pivot3)
                st.subheader("Total amount required for loan based on Loan Grade")
                st.dataframe(pivot4)
                st.subheader("Total amount required for loan based on Loan intent and Home Ownership")
                st.dataframe(pivot5)
                st.subheader("Total amount required for loan based on Loan intent and Loan Grade")
                st.dataframe(pivot6)
                st.subheader("Total amount required for loan based on Home Ownership and Loan Grade")
                st.dataframe(pivot7)
                st.subheader("Total amount required for loan")
                st.dataframe(pivot1)
                st.subheader("Predicated data for each customer")
                st.dataframe(dfp)
                
                st.header("Graph")
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='person_age', hue='loan status', data=dfp)
                plt.title('person_age')
                st.pyplot(fig)
    
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='person_home_ownership', hue='loan status', data=dfp)
                plt.title('person_home_ownership')
                st.pyplot(fig)
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='loan_intent', hue='loan status', data=dfp)
                plt.title('loan_intent')
                st.pyplot(fig)
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='loan_grade', hue='loan status', data=dfp)
                plt.title('loan_grade')
                st.pyplot(fig)
            if reportresult=='Loan Status':
                st.header("Data related to Loan Status")
                st.subheader("**Total number of customer applied for loan are as follows**")
                st.dataframe(dfp.groupby('loan status')['loan status'].count())
                st.subheader("Total amount required for loan")
                st.dataframe(pivot)
                st.subheader("**Loan intent for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('loan_intent')['loan status'].count())
                st.subheader("**Loan intent for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('loan_intent')['loan status'].count())
                st.subheader("**Home Ownership for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('person_home_ownership')['loan status'].count())
    
                st.subheader("**Home Ownership for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('person_home_ownership')['loan status'].count())
                st.subheader("**Loan Grade for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('loan_grade')['loan status'].count())
                st.subheader("**Loan Grade for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('loan_grade')['loan status'].count())
                st.subheader("Total amount required for loan")
                st.dataframe(pivot1)
 
                
                st.header("Graph")
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='person_age', hue='loan status', data=dfp)
                plt.title('person_age')
                st.pyplot(fig)
    
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='person_home_ownership', hue='loan status', data=dfp)
                plt.title('person_home_ownership')
                st.pyplot(fig)
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='loan_intent', hue='loan status', data=dfp)
                plt.title('loan_intent')
                st.pyplot(fig)
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='loan_grade', hue='loan status', data=dfp)
                plt.title('loan_grade')
                st.pyplot(fig)
            if reportresult=='Loan Intent':
                st.header("Data related to Loan Intent")
                st.subheader("**Loan intent for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('loan_intent')['loan status'].count())
                st.subheader("**Loan intent for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('loan_intent')['loan status'].count())
    
                st.subheader("Total amount required for loan based on Loan intent")
                st.dataframe(pivot2)


                st.subheader("Total amount required for loan based on Loan intent and Home Ownership")
                st.dataframe(pivot5)
                st.subheader("Total amount required for loan based on Loan intent and Loan Grade")
                st.dataframe(pivot6)
                st.subheader("Total amount required for loan")
                st.dataframe(pivot1)
                
                st.header("Graph")
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='loan_intent', hue='loan status', data=dfp)
                plt.title('loan_intent')
                st.pyplot(fig)
            if reportresult=='Home Ownership':
                st.header("Data related to Home Ownership")

                st.subheader("**Home Ownership for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('person_home_ownership')['loan status'].count())
    
                st.subheader("**Home Ownership for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('person_home_ownership')['loan status'].count())
                st.subheader("Total amount required for loan based on Home Ownership")
                st.dataframe(pivot3)
                st.subheader("Total amount required for loan based on Loan intent and Home Ownership")
                st.dataframe(pivot5)
                st.subheader("Total amount required for loan based on Home Ownership and Loan Grade")
                st.dataframe(pivot7)
                st.subheader("Total amount required for loan")
                st.dataframe(pivot1)
                
                st.header("Graph")
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='person_home_ownership', hue='loan status', data=dfp)
                plt.title('person_home_ownership')
                st.pyplot(fig)     
            if reportresult=='Loan grade':
                st.header("Data related to Loan grade")
                st.subheader("**Loan Grade for Not a defaulter customer are as follows**")
                st.dataframe(dfp1.groupby('loan_grade')['loan status'].count())
                st.subheader("**Loan Grade for Defaulter customer are as follows**")
                st.dataframe(dfp2.groupby('loan_grade')['loan status'].count())
                st.subheader("Total amount required for loan based on Loan Grade")
                st.dataframe(pivot4)
                st.subheader("Total amount required for loan based on Loan intent and Loan Grade")
                st.dataframe(pivot6)
                st.subheader("Total amount required for loan based on Home Ownership and Loan Grade")
                st.dataframe(pivot7)
                st.subheader("Total amount required for loan")
                st.dataframe(pivot1)

                st.header("Graph")
                fig = plt.figure(figsize=(20, 10))
                sns.countplot(x='loan_grade', hue='loan status', data=dfp)
                plt.title('loan_grade')
                st.pyplot(fig)                
            
