__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

from sklearn.preprocessing import MinMaxScaler

import category_encoders
import fputils

'''
This file implements the minmax scaler for challenge 2 which is to scale the numerical features.
'''

def get_minmax_scaling_results():
    print('Performing minmax scaling...')
    print()
    columns_to_use = "Gender,Senior Citizen,Partner,Dependents,Tenure Months," \
                     "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection," \
                     "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method,Monthly Charges" \
                     ",Total Charges,Churn Value".split(',')

    telco_df = fputils.load_data(columns_to_use)

    def perform_encoding(df):
        columns_for_encoding = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                                'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
                                'Tech Support',
                                'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
        encoder = category_encoders.CountEncoder(cols=columns_for_encoding)
        return encoder.fit_transform(df)

    telco_df_encoded = perform_encoding(telco_df)

    def perform_scaling(df):
        columns_to_scale = ["Tenure Months", "Monthly Charges", "Total Charges"]
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    telco_df_scaled = perform_scaling(telco_df_encoded)

    X, y = fputils.get_X_and_y(telco_df_scaled)

    fputils.train_classifiers(X, y)
