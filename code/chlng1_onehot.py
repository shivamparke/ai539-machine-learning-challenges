__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

import category_encoders
import fputils

'''
This file implements the onehot encoding for challenge 1 which is to encode categorical data.
'''

def get_onehot_results():
    print('Performing one-hot encoding...')
    print()
    columns_to_use = "Gender,Senior Citizen,Partner,Dependents,Tenure Months," \
                     "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection," \
                     "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method,Monthly Charges" \
                     ",Total Charges,Churn Value".split(',')
    # Loads the dataset into a dataframe
    telco_df = fputils.load_data(columns_to_use)

    def perform_encoding(df):
        columns_for_encoding = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                                'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
                                'Tech Support',
                                'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
        encoder = category_encoders.OneHotEncoder(cols=columns_for_encoding)
        return encoder.fit_transform(df)

    # This is the newly encoded dataframe
    telco_df_encoded = perform_encoding(telco_df)
    # Get the feature and label vectors
    X, y = fputils.get_X_and_y(telco_df_encoded)
    # Train and get the results
    fputils.train_classifiers(X, y)
