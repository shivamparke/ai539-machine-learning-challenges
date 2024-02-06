__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders
import fputils

'''
This file implements the replace location related info by x,y,z coordinates strategy to represent location related information for challenge 4.
'''

def get_replace_with_xyz_results():
    print('Transforming location information into (x,y,z) co-ordinates...')
    print()
    columns_to_use = "Latitude,Longitude,Gender,Senior Citizen,Partner,Dependents,Tenure Months," \
                     "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection," \
                     "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method,Monthly Charges" \
                     ",Total Charges,Churn Value".split(',')

    telco_df = fputils.load_data(columns_to_use)
    # Extract the values from the df as 1 column
    lat = telco_df['Latitude'].values
    lon = telco_df['Longitude'].values

    # map the latitude and longitude values to x, y, and z coordinates
    x = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = np.sin(np.radians(lat))

    # create a new DataFrame with the x, y, and z coordinates
    df_xyz = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # perform standardization on the x, y, and z coordinates
    scaler = MinMaxScaler()
    df_xyz_scaled = scaler.fit_transform(df_xyz)

    # insert the new columns at the index of the "Latitude" column
    telco_df.insert(telco_df.columns.get_loc("Latitude"), "x", df_xyz_scaled[:, 0])
    telco_df.insert(telco_df.columns.get_loc("Latitude"), "y", df_xyz_scaled[:, 1])
    telco_df.insert(telco_df.columns.get_loc("Latitude"), "z", df_xyz_scaled[:, 2])

    # drop the original Latitude and Longitude columns
    telco_df.drop(['Latitude', 'Longitude'], axis=1, inplace=True)


    def perform_encoding(df):
        columns_for_encoding = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                                'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
                                'Tech Support',
                                'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
        encoder = category_encoders.CountEncoder(cols=columns_for_encoding)
        return encoder.fit_transform(df)

    telco_df_encoded = perform_encoding(telco_df)

    X, y = fputils.get_X_and_y(telco_df_encoded)

    fputils.train_classifiers(X, y)
