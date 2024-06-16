"""
Created on Thu May 23 19:34:56 2024

@author: Krishna Salve
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All'

@app.route('/predict')
def predict_loan_approval():
    income_annum=request.args.get('income_annum')
    loan_amount=request.args.get('loan_amount')
    loan_term=request.args.get('loan_term')
    cibil_score=request.args.get('cibil_score')
    commercial_assets_value=request.args.get('commercial_assets_value')
    prediction=classifier.predict([[income_annum,loan_amount,loan_term,cibil_score,commercial_assets_value]])
    return 'The predicted value is ' + str(prediction)


@app.route('/predict_file',methods=['POST'])
def predict_test_loan_approval():
    print("exec 0")
    df_test=pd.read_csv(request.files.get('file'))
    print("exec 1")
    print(df_test)
    print("exec 2")
    dtest = df_test.values
    print(dtest)
    #dtest = xgb.DMatrix(np.asmatrix(X_test), label=y_test)
    prediction=classifier.predict(dtest)
    return 'The predicted values for the csv is ' + str(list(prediction))

if __name__ == '__main__':
    app.debug=True
    app.run()