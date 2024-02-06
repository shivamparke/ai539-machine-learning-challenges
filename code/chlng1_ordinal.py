__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

import category_encoders
import fputils

'''
This file implements the ordinal + one-hot encoding for challenge 1 which is to encode categorical data.
'''


def get_ordinal_encoding_results():
    print('Performing one-hot and ordinal encoding...')
    print()
    columns_to_use = "Gender,Senior Citizen,Partner,Dependents,Tenure Months," \
                     "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection," \
                     "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method,Monthly Charges" \
                     ",Total Charges,Churn Value".split(',')
    # Loads the dataset into a dataframe
    telco_df = fputils.load_data(columns_to_use)

    def perform_encoding(df):
        columns_for_encoding = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                                'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                                'Streaming TV', 'Streaming Movies', 'Paperless Billing', 'Payment Method']
        mappings = [
            {"col": "Internet Service", "mapping": {"No": 0, "DSL": 1, "Fiber optic": 2}},
            {"col": "Contract", "mapping": {"Month-to-month": 0, "One year": 1, "Two year": 2}}
        ]
        ordinal_columns = ['Internet Service', 'Contract']
        encoder = category_encoders.OneHotEncoder(cols=columns_for_encoding)
        encoder2 = category_encoders.OrdinalEncoder(mapping=mappings, cols=ordinal_columns)
        telco_df_encoded2 = encoder.fit_transform(telco_df)
        return encoder2.fit_transform(telco_df_encoded2)

    # This is the newly encoded dataframe
    telco_df_encoded = perform_encoding(telco_df)
    # Get the feature and label vectors
    X, y = fputils.get_X_and_y(telco_df_encoded)
    # Train and get the results
    fputils.train_classifiers(X, y)
