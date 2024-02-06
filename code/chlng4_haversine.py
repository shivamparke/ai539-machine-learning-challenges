__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

from math import radians, sin, cos, sqrt, atan2
import category_encoders
from sklearn.preprocessing import MinMaxScaler

import fputils

'''
This file implements the haversine distance strategy to represent location related information for challenge 4.
'''

def get_haversine_strategy_results():
    print('Transforming location information into haversine distances...')
    print()
    columns_to_use = "Latitude,Longitude,Gender,Senior Citizen,Partner,Dependents,Tenure Months," \
                     "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection," \
                     "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method,Monthly Charges" \
                     ",Total Charges,Churn Value".split(',')

    telco_df = fputils.load_data(columns_to_use)

    # Define a function to calculate the Haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the earth in km
        # The formula as found on: https://www.igismap.com/haversine-formula-calculate-geographic-distance-earth/
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        a = sin(dLat / 2) * sin(dLat / 2) + cos(radians(lat1)) \
            * cos(radians(lat2)) * sin(dLon / 2) * sin(dLon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    # Define the center of Los Angeles, CA as fixed latitude and longitude value
    center_lat = 34.0522
    center_lon = -118.2437

    # Calculate the Haversine distance for each data point
    telco_df.insert(loc=telco_df.columns.get_loc("Latitude"), column="Haversine Distance",
                    value=telco_df.apply(
                        lambda row: haversine_distance(center_lat, center_lon, row['Latitude'], row['Longitude']),
                        axis=1))

    # Drop the Latitude and Longitude columns
    telco_df = telco_df.drop(['Latitude', 'Longitude'], axis=1)

    def perform_encoding(df):
        columns_for_encoding = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                                'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
                                'Tech Support',
                                'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
        encoder = category_encoders.CountEncoder(cols=columns_for_encoding)
        return encoder.fit_transform(df)

    telco_df_encoded = perform_encoding(telco_df)

    def perform_scaling(df):
        columns_to_scale = ["Haversine Distance", "Tenure Months", "Monthly Charges", "Total Charges"]
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    telco_df_scaled = perform_scaling(telco_df_encoded)

    X, y = fputils.get_X_and_y(telco_df_scaled)

    fputils.train_classifiers(X, y)
