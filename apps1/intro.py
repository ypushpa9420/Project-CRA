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
   st.header("How to use this tool")
   st.write("1.	Choose Single prediction or Bulk prediction from dropdown")
   st.write("2.	For single prediction fill the customer’s data and click on “Submit” button from left panel and see the result in right panel of this tool")
   st.write("3.	For multiple prediction, upload bulk data of customers from left panel and see the result in right panel of this tool")
   st.header("Benefit of using this tool")
   st.write("1.	This tool help to predict loan defaulter based on data present to banks/finance company")
   st.write("2.	Apart from future prediction this tool is capable of giving analysis of data of customer who have applied for loan like number of customer apply for loan, which customer is defaulter which one is not, what kind of loan required and what amount of load is required etc.")
   st.write("3. This tool help company to identify customer whom they can provide loan, amount of loan and type of loan")

    
    
   

