import sys

# setting the random seed for reproducibility
import random
random.seed(493)

# for manipulating dataframes
import pandas as pd
import numpy as np

# for modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import r2_score

def dummify(df, column):
    """
    Takes a dataframe and column to return a dataframe with 
    dummy variables appended.
    """
    dummy = pd.get_dummies(df[column], prefix=column, prefix_sep='_',)
    return pd.concat([df, dummy], axis=1)

def main():
    """Main entry point for the script."""

    # read the csv file
    df = pd.read_csv('churn_clean.csv')

    # fill missing values with None as in no service
    df = df.fillna("None")

    # drop columns
    df.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County',
                    'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 'Job',
                    'Churn', 'Income', 'PaperlessBilling', 'PaymentMethod',
                    'Item1', 'Item2', 'Item3', 'Item4', 'Item5',
                    'Item6', 'Item7', 'Item8'], inplace=True)

    # assemble list of categorical columns to generate dummy variables for
    dummy_columns = ['Marital',
                    'Gender',
                    'Techie',
                    'Contract',
                    'Port_modem',
                    'Tablet',
                    'InternetService',
                    'Phone',
                    'Multiple',
                    'OnlineSecurity',
                    'OnlineBackup',
                    'DeviceProtection',
                    'TechSupport',
                    'StreamingTV',
                    'StreamingMovies'
                    ]

    dummified = df.copy()

    # loop through all the columns tp generate dummy for
    for col in dummy_columns:
        dummified = dummify(dummified, col)


    # drop original columns we generated dummies for
    dummified.drop(columns=dummy_columns, inplace=True)

    # move target variable at the end of the dataframe
    df = dummified[['Children', 'Age', 'Outage_sec_perweek', 'Email', 'Contacts',
        'Yearly_equip_failure', 'Tenure', 'Bandwidth_GB_Year',
        'Marital_Divorced', 'Marital_Married', 'Marital_Never Married',
        'Marital_Separated', 'Marital_Widowed', 'Gender_Female', 'Gender_Male',
        'Gender_Nonbinary', 'Techie_No', 'Techie_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two Year',
        'Port_modem_No', 'Port_modem_Yes', 'Tablet_No', 'Tablet_Yes',
        'InternetService_DSL', 'InternetService_Fiber Optic',
        'InternetService_None', 'Phone_No', 'Phone_Yes', 'Multiple_No',
        'Multiple_Yes', 'OnlineSecurity_No', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_Yes', 'DeviceProtection_No',
        'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_Yes', 'StreamingMovies_No',
        'StreamingMovies_Yes', 'MonthlyCharge']]

    # replace True with 1's and False with 0's
    df = df.replace(True, 1)
    df = df.replace(False, 0)

    scaler = MinMaxScaler()

    # Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
    num_vars = ['Children', 'Age', 'Outage_sec_perweek', 'Email', 'Contacts','Yearly_equip_failure', 'Tenure', 'Bandwidth_GB_Year', 'MonthlyCharge']
    df[num_vars] = scaler.fit_transform(df[num_vars])

    # split the dataframe between independent and dependent variables
    X = df.drop('MonthlyCharge',axis= 1)
    y = df[['MonthlyCharge']]

    # split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=493)

    rfe_columns = ['Children', 'Age', 'Tenure', 'Bandwidth_GB_Year']

    # Creating X_test dataframe with RFE-selected variables
    X_train_rfe = X_train[rfe_columns]

    # Adding a constant variable
    X_train_rfe = sm.add_constant(X_train_rfe)

    lm = sm.OLS(y_train,X_train_rfe).fit()   # run the linear model

    print(lm.summary())

    # make predictions.

    X_train_rfe = X_train_rfe.drop(['const'], axis=1)

    # Create X_test_new dataframe by dropping variables from X_test
    X_test_rfe = X_test[X_train_rfe.columns]

    # add a constant variable 
    X_test_rfe = sm.add_constant(X_test_rfe)

    # make predictions
    y_pred = lm.predict(X_test_rfe)

    r2_score(y_true = y_test, y_pred = y_pred)

    print("R2 score: " + str(r2_score(y_true = y_test, y_pred = y_pred)))

if __name__ == '__main__':
    sys.exit(main())