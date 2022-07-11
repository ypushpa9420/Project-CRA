# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:00:42 2021

@author: Pushpa Yadav
"""
# import streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def app():
    st.header('EDA and VDA')
    df= pd.read_pickle("./finaldataframe.pkl")
    #df1= pd.read_pickle("./finaldataframe.pkl")
    dfo = pd.read_csv('./credit_risk_dataset.csv')
    st.subheader('**Below is the structure of dataset**')
    oBuffer = io.StringIO()
    dfo.info(buf=oBuffer)
    vBuffer = oBuffer.getvalue()
    st.text(vBuffer)    
    #st.write(df.columns)
            
    # columns
    st.subheader('**Below is the columns of dataset**')
    st.text(dfo.columns)    
    
    df['person_home_ownership']=df['person_home_ownership'].map({0:'MORTGAGE',1:'OTHER',2:'OWN',3:'RENT'})

    df['loan_intent']=df['loan_intent'].map({0:'DEBTCONSOLIDATION',1:'EDUCATION',2:'HOMEIMPROVEMENT',3:'MEDICAL',4:'PERSONAL',5:'VENTURE'})

    df['loan_grade']=df['loan_grade'].map({0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'})


    df['cb_person_default_on_file']=df['cb_person_default_on_file'].map({0:'N',1:'Y'})

    df['loan_status']=df['loan_status'].map({0:"Not a defaulter",1:'Defaulter'})
    st.subheader("**This is our cleaned dataset**")
    oBuffer1 = io.StringIO()
    df.info(buf=oBuffer1)
    vBuffer1 = oBuffer1.getvalue()
    st.text(vBuffer1)    
    #st.write(df.columns)
    st.dataframe(df)
    csv = convert_df(df)
    st.download_button(label="Download Data",data=csv,file_name='dfp.csv',mime='text/csv')

    fig = plt.figure(figsize=(20, 10))
    sns.countplot(x='person_age', hue='loan_status', data=df)
    plt.title('Age VS Loan status')
    st.subheader("**We can observe that people who are younger have a tendency not to pay the loan, 0 paid and 1 did not. The greatest default is among the youngest.**")
    st.pyplot(fig)
    st.subheader("**In this graph we can analyze the what were the reasos for the loan.**")
    fig = plt.figure(figsize=(20, 10))
    #sns.countplot(defaulterv['loan_intent'])
    sns.countplot(x='loan_status', hue='loan_intent', data=df)
    plt.title('Loan Intent')
    st.pyplot(fig)
    fig = plt.figure(figsize=(20, 10))
    #sns.countplot(defaulterv['loan_intent'])
    st.subheader("**In this graph we can analyze the Loan grade based on their loan intent**")
    sns.countplot(x='loan_grade', hue='loan_intent', data=df)
    plt.title('Loan Grade')
    st.pyplot(fig)
    
  
 
