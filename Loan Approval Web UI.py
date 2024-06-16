"""
Created on Thu May 23 19:34:56 2024

@author: Krishna Salve
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger # type: ignore
from flasgger import Swagger # type: ignore

app=Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All'

@app.route('/predict', methods=['Get'])
def predict_loan_approval():
    
    """let's Authenticate the Loan Approval
    This is using docstrings for specifications.
    ---
    parameters:
        - name: income_annum
          in: query
          type: number
          required: true
        - name: loan_amount
          in: query
          type: number
          required: true
        - name: loan_term
          in: query
          type: number
          required: true
        - name: cibil_score
          in: query
          type: number
          required: true
        - name: commercial_assets_value
          in: query
          type: number
          required: true
    responses:
        200: 
            description: The output values
            
    """
        
        
    income_annum=request.args.get('income_annum')
    loan_amount=request.args.get('loan_amount')
    loan_term=request.args.get('loan_term')
    cibil_score=request.args.get('cibil_score')
    commercial_assets_value=request.args.get('commercial_assets_value')
    prediction=classifier.predict([[income_annum,loan_amount,loan_term,cibil_score,commercial_assets_value]])
    return 'The predicted value is ' + str(prediction)


@app.route('/predict_file',methods=['POST'])
def predict_test_loan_approval():
    """Let's Authenticate the Loan Approval
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    
    responses:
        200:
            description: The output values
            
    """
    df_test=pd.read_csv(request.files.get('file'))
    dtest = df_test.values
    prediction=classifier.predict(dtest)
    return  str(list(prediction))

if __name__ == '__main__':
    app.run()