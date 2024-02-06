__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

'''
This file contains the common functions executed by all the different strategies for each challenge.
'''

import numpy as np
import pandas as pd
import category_encoders
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Define the seed
seed = 0

# Will be used only for output printing
classfier_names = {0: "KNN", 1: "Baseline"}


def load_data(columns_to_use):
    '''
    This function loads the dataset into a Pandas dataframe.
    :param columns_to_use: the columns which we want to load.
    :return: a Pandas dataframe.
    '''
    # Define how to handle missing values
    missing_vals = ['', ' ']
    # Read all the rows of the above columns from the XLSX file
    telco_df = pd.read_excel(io="Telco_customer_churn.xlsx", na_values=missing_vals, usecols=columns_to_use)
    # Preprocessing step for customers who have been using service for less than a month
    # Their "Total Charges" column is empty; we replace it by the value in their "Monthly Charges" column
    telco_df['Total Charges'].fillna(telco_df['Monthly Charges'], inplace=True)
    return telco_df


def get_X_and_y(telco_df_encoded):
    '''
    This function gets us the feature and label vectors.
    :param telco_df_encoded: the encoded dataframe without any categorical data.
    :return: The feature vector X and the label vector y
    '''
    numpy_telco = telco_df_encoded.to_numpy()
    total_columns = numpy_telco.shape
    X = numpy_telco[:, 0:total_columns[1] - 1]
    # print(X.shape)
    y = numpy_telco[:, total_columns[1] - 1:]
    y = y.astype('int')
    return X, y


def train_classifiers(X, y):
    '''
    This function trains the KNN and baseline classifier.
    :param X: the feature vector
    :param y: the label vector
    :return: None. It only prints the results of trained classifiers.
    '''
    classifiers = []
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    classifiers.append(knn_classifier)
    baseline_classifier = DummyClassifier(strategy="stratified", random_state=seed)
    classifiers.append(baseline_classifier)
    cv_stratified10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for index, classifier in enumerate(classifiers):
        accuracy_scores = cross_val_score(classifier, X, y.ravel(), scoring='accuracy', cv=cv_stratified10fold,
                                          n_jobs=-1)
        average_accuracy = round(np.mean(accuracy_scores) * 100, 2)
        precision_scores = cross_val_score(classifier, X, y.ravel(), scoring='precision', cv=cv_stratified10fold,
                                           n_jobs=-1)
        average_precision = round(np.mean(precision_scores) * 100, 2)
        recall_scores = cross_val_score(classifier, X, y.ravel(), scoring='recall', cv=cv_stratified10fold, n_jobs=-1)
        average_recall = round(np.mean(recall_scores) * 100, 2)
        f2_score = (5 * average_precision * average_recall) / ((4 * average_precision) + average_recall)
        print(f"Average {classfier_names[index]} accuracy:", average_accuracy)
        print(f"Average {classfier_names[index]} precision:", average_precision)
        print(f"Average {classfier_names[index]} recall:", average_recall)
        print(f"{classfier_names[index]} F2 score:", f2_score)
        print()


def tune_hyperparameters():
    '''
    This function performs hyperparameter tuning by using the best strategies for each challenge.
    :return: a dictionary of the optimal hyperparameters for the KNN classifier.
    '''
    seed = 0

    columns_to_use = "Gender,Senior Citizen,Partner,Dependents,Tenure Months," \
                     "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection," \
                     "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method,Monthly Charges" \
                     ",Total Charges,Churn Value".split(',')

    telco_df = load_data(columns_to_use)

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

    X, y = get_X_and_y(telco_df_scaled)

    over_sampler = RandomOverSampler(random_state=seed)
    X_balanced, y_balanced = over_sampler.fit_resample(X, y)
    X = X_balanced
    y = y_balanced

    # define the KNN classifier object
    knn = KNeighborsClassifier()
    # Results:
    # {'algorithm': 'ball_tree', 'leaf_size': 10, 'metric': 'manhattan', 'n_jobs': -1, 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}

    # define the hyperparameter grid to search over
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2],
        'n_jobs': [-1]  # use all available CPU cores for parallel processing
    }

    # perform grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(knn, param_grid, cv=10, n_jobs=-1)
    grid_search.fit(X, y.ravel())

    return grid_search.best_params_
