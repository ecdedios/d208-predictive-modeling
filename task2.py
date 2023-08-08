import sys

# setting the random seed for reproducibility
import random
random.seed(493)

# for manipulating dataframes
import pandas as pd
import numpy as np

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# for modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics

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
    df.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
        'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 'Job',
            'Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8'],
            inplace=True)

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
                    'StreamingMovies',
                    'PaperlessBilling',
                    'PaymentMethod'
                    ]

    dummified = df.copy()

    # loop through all the columns tp generate dummy for
    for col in dummy_columns:
        dummified = dummify(dummified, col)

    # drop original columns we generated dummies for
    dummified.drop(columns=dummy_columns, inplace=True)

    # move target variable at the end of the dataframe
    df = dummified[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email',
        'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge',
        'Bandwidth_GB_Year', 'Marital_Divorced', 'Marital_Married',
        'Marital_Never Married', 'Marital_Separated', 'Marital_Widowed',
        'Gender_Female', 'Gender_Male', 'Gender_Nonbinary', 'Techie_No',
        'Techie_Yes', 'Contract_Month-to-month', 'Contract_One year',
        'Contract_Two Year', 'Port_modem_No', 'Port_modem_Yes', 'Tablet_No',
        'Tablet_Yes', 'InternetService_DSL', 'InternetService_Fiber Optic',
        'InternetService_None', 'Phone_No', 'Phone_Yes', 'Multiple_No',
        'Multiple_Yes', 'OnlineSecurity_No', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_Yes', 'DeviceProtection_No',
        'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_Yes', 'StreamingMovies_No',
        'StreamingMovies_Yes', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank Transfer(automatic)',
        'PaymentMethod_Credit Card (automatic)',
        'PaymentMethod_Electronic Check', 'PaymentMethod_Mailed Check', 'Churn']]

    # replace True with 1's and False with 0's
    df = df.replace(True, 1)
    df = df.replace(False, 0)

    # replace 'Yes' with 1's and 'No' with 0's
    df['Churn'] = df['Churn'].replace('Yes', 1)
    df['Churn'] = df['Churn'].replace('No', 0)

    scaler = MinMaxScaler()

    # apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
    num_vars = ['Children', 'Age', 'Outage_sec_perweek', 'Email', 'Contacts','Yearly_equip_failure', 'Tenure', 'Bandwidth_GB_Year', 'MonthlyCharge']
    df[num_vars] = scaler.fit_transform(df[num_vars])

    # split the dataframe between independent and dependent variables
    X = df.drop('Churn',axis= 1)
    y = df[['Churn']]

    # split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=493)

    sm = SMOTE(random_state=493)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    logreg = LogisticRegression()
    logreg.fit(X_res, y_res.values.ravel())

    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    cfm = confusion_matrix(y_test, y_pred)

    print(cfm)
    print(classification_report(y_test, y_pred))

    rfe_columns = ['StreamingMovies_No', 'Tenure', 'StreamingTV_No', 'Bandwidth_GB_Year', 'Multiple_No']

    # create dataframe with RFE-selected variables
    X_res_rfe = X_res[rfe_columns]

    # create dataframe with RFE-selected variables
    X_test_rfe = X_test[rfe_columns]

    logreg2 = LogisticRegression()
    logreg2.fit(X_res_rfe, y_res.values.ravel())

    y_pred2 = logreg2.predict(X_test_rfe)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg2.score(X_test_rfe, y_test)))

    cfm2 = confusion_matrix(y_test, y_pred2)

    print(cfm2)
    print(classification_report(y_test, y_pred2))

    #define metrics
    y_pred_proba = logreg2.predict_proba(X_test_rfe)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    print("AUC Score: " + str(auc))

if __name__ == '__main__':
    sys.exit(main())