# import streamlit
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        dfp['Load status']=dfp['Predict'].map({0:"Not a defaulter",1:'Defaulter'})
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
            #st.subheader('Input Data')
            #st.dataframe(dfp)
            # predict
            dfp = getPredict(dfp)
            # show dataframe
            #st.subheader('Prediction')
           
            # show accuracy
# =============================================================================
#             if bClsVars:
#                 st.subheader('Accuracy')
#                 from sklearn.metrics import accuracy_score
#                 accuracy = accuracy_score(dfp[clsVars], dfp['Predict'])*100
#                 st.text(accuracy)
# =============================================================================
            dfp['person_home_ownership']=dfp['person_home_ownership'].map({0:'MORTGAGE',1:'OTHER',2:'OWN',3:'RENT'})
            dfp['loan_intent']=dfp['loan_intent'].map({0:'DEBTCONSOLIDATION',1:'EDUCATION',2:'HOMEIMPROVEMENT',3:'MEDICAL',4:'PERSONAL',5:'VENTURE'})

            dfp['loan_grade']=dfp['loan_grade'].map({0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'})

            dfp['cb_person_default_on_file']=dfp['cb_person_default_on_file'].map({0:'N',1:'Y'})
            csv = convert_df(dfp)
            filter1=dfp['Load status']=="Not a defaulter"
            dfp1=dfp.where(filter1)
            filter2=dfp['Load status']=="Defaulter"
            dfp2=dfp.where(filter2)
            st.subheader("Thanks for using this tool. Please see below analysis and prediction for your customer data")
            st.subheader("**Total number of customer applied for loan are as follows**")
            st.dataframe(dfp.groupby('Load status')['Load status'].count())
            csv1 = convert_df(dfp.groupby('Load status')['Load status'].count())
            #st.download_button(label="Download Loan status",data=csv1,file_name='dfp.csv',mime='text/csv')
            pivot = dfp.pivot_table(index =['Load status'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan")
            st.dataframe(pivot)
            #st.download_button(label="Download Total amount required for loan",data=csv,file_name='dfp.csv',mime='text/csv')
            st.subheader("**Loan intent for Not a defaulter customer are as follows**")
            st.dataframe(dfp1.groupby('loan_intent')['Load status'].count())
            #st.download_button(label="Download Loan intent for Not a defaulter customer",data=csv,file_name='dfp.csv',mime='text/csv')
            st.subheader("**Loan intent for Defaulter customer are as follows**")
            st.dataframe(dfp2.groupby('loan_intent')['Load status'].count())
            #st.download_button(label="Download Loan intent for Defaulter customer",data=csv,file_name='dfp.csv',mime='text/csv')
            st.subheader("**Home Ownership for Not a defaulter customer are as follows**")
            st.dataframe(dfp1.groupby('person_home_ownership')['Load status'].count())
            #st.download_button(label="Download Loan intent for Defaulter customer",data=csv,file_name='dfp.csv',mime='text/csv')

            st.subheader("**Home Ownership for Defaulter customer are as follows**")
            st.dataframe(dfp2.groupby('person_home_ownership')['Load status'].count())
            st.subheader("**Loan Grade for Not a defaulter customer are as follows**")
            st.dataframe(dfp1.groupby('loan_grade')['Load status'].count())
            st.subheader("**Loan Grade for Defaulter customer are as follows**")
            st.dataframe(dfp2.groupby('loan_grade')['Load status'].count())
            pivot2 = dfp.pivot_table(index =['Load status','loan_intent'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan based on Loan intent")
            st.dataframe(pivot2)
            pivot3 = dfp.pivot_table(index =['Load status','person_home_ownership'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan based on Home Ownership")
            st.dataframe(pivot3)
            pivot4 = dfp.pivot_table(index =['Load status','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan based on Home Ownership")
            st.dataframe(pivot4)
            pivot5 = dfp.pivot_table(index =['Load status','loan_intent','person_home_ownership'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan based on Loan intent and Home Ownership")
            st.dataframe(pivot5)
            pivot6 = dfp.pivot_table(index =['Load status','loan_intent','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan based on Loan intent and Loan Grade")
            st.dataframe(pivot6)
            pivot7 = dfp.pivot_table(index =['Load status','person_home_ownership','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan based on Home Ownership and Loan Grade")
            st.dataframe(pivot7)
            pivot1 = dfp.pivot_table(index =['Load status','loan_intent','person_home_ownership','loan_grade'],values =['loan_amnt'], aggfunc ='sum')
            st.subheader("Total amount required for loan")
            st.dataframe(pivot1)
            st.subheader("Predicated data for each customer")
            st.dataframe(dfp)
            #st.download_button(label="Download Data",data=csv,file_name='dfp.csv',mime='text/csv')
            
           
            fig = plt.figure(figsize=(20, 10))
            sns.countplot(x='person_age', hue='Load status', data=dfp)
            plt.title('person_age')
            st.pyplot(fig)

            fig = plt.figure(figsize=(20, 10))
            sns.countplot(x='person_home_ownership', hue='Load status', data=dfp)
            plt.title('person_home_ownership')
            st.pyplot(fig)
            fig = plt.figure(figsize=(20, 10))
            sns.countplot(x='loan_intent', hue='Load status', data=dfp)
            plt.title('loan_intent')
            st.pyplot(fig)
            fig = plt.figure(figsize=(20, 10))
            sns.countplot(x='loan_grade', hue='Load status', data=dfp)
            plt.title('loan_grade')
            st.pyplot(fig)


            if (st.button('Reset')):
                vFileDets = None
