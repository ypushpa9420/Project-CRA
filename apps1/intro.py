# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:00:42 2021

@author: Pushpa Yadav
"""
# import streamlit
import streamlit as st

    

def app():
    
   st.header("**Introduction**")
   st.write("Credit default risk is simply known as the possibility of a loss for a lender due to a borrower’s failure to repay a loan. Credit analysts are typically responsible for assessing this risk by thoroughly analyzing a borrower’s capability to repay a loan — but long gone are the days of credit analysts, it’s the machine learning age! Machine learning algorithms have a lot to offer to the world of credit risk assessment due to their unparalleled predictive power and speed. In this article, we will be utilizing machine learning’s power to predict whether a borrower will default on a loan or not and to predict their probability of default. Let’s get started.")
   st.header("**Dataset**")
   st.write("The dataset we’re using can be found on Kaggle and it contains data for 32,581 borrowers and 11 variables related to each borrower. Let’s have a look at what those variables are:")

   st.write("Age - numerical variable; age in years")
   st.write("Income - numerical variable; annual income in dollars")
   st.write("Home status - categorical variable; rent, mortgage or own")
   st.write("Employment length - numerical variable; employment length in years")
   st.write("Loan intent - categorical variable; education, medical, venture, home improvement, personal or debt consolidation")
   st.write("Loan amount - numerical variable; loan amount in dollars")
   st.write("Loan grade - categorical variable; A, B, C, D, E, F or G")
   st.write("Interest rate - numerical variable; interest rate in percentage")
   st.write("Loan to income ratio - numerical variable; between 0 and 1")
   st.write("Historical default - binary, categorical variable; Y or N")
   st.write("Loan status - binary, numerical variable; 0 (no default) or 1 (default) - this is going to be our target variable)")
