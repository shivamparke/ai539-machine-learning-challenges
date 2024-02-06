__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

import numpy as np
from imblearn.over_sampling import RandomOverSampler
import category_encoders
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import fputils

'''
This file performs random oversampling to solve the class imbalance challenge 3.
'''

seed = 0

def get_random_oversampling_results():
    print('Performing random oversampling...')
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

    X, y = fputils.get_X_and_y(telco_df_encoded)
    # Saved for use later
    X_old = X
    y_old = y

    # Perform random oversampling
    over_sampler = RandomOverSampler(random_state=seed)
    X_balanced, y_balanced = over_sampler.fit_resample(X, y)
    X_new = X_balanced
    y_new = y_balanced
    # num_oversampled_rows = len(over_sampler.sample_indices_) - len(X)
    # print("Number of oversampled rows:", num_oversampled_rows)
    # print(X_new.shape)

    # fputils.train_classifiers(X, y)

    classifiers = []
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    classifiers.append(knn_classifier)
    baseline_classifier = DummyClassifier(strategy="stratified", random_state=seed)
    classifiers.append(baseline_classifier)
    cv_stratified10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for index, classifier in enumerate(classifiers):
        if index == 0:
            # If the classifier is KNN, use the balanced dataset
            X = X_new
            y = y_new
        else:
            # If the classifier is baseline, use the old dataset as told in the feedback
            X = X_old
            y = y_old
        accuracy_scores = cross_val_score(classifier, X, y.ravel(), scoring='accuracy', cv=cv_stratified10fold,
                                          n_jobs=-1)
        average_accuracy = round(np.mean(accuracy_scores) * 100, 2)
        precision_scores = cross_val_score(classifier, X, y.ravel(), scoring='precision', cv=cv_stratified10fold,
                                           n_jobs=-1)
        average_precision = round(np.mean(precision_scores) * 100, 2)
        recall_scores = cross_val_score(classifier, X, y.ravel(), scoring='recall', cv=cv_stratified10fold, n_jobs=-1)
        average_recall = round(np.mean(recall_scores) * 100, 2)
        f2_score = (5 * average_precision * average_recall) / ((4 * average_precision) + average_recall)
        print(f"Average {fputils.classfier_names[index]} accuracy:", average_accuracy)
        print(f"Average {fputils.classfier_names[index]} precision:", average_precision)
        print(f"Average {fputils.classfier_names[index]} recall:", average_recall)
        print(f"{fputils.classfier_names[index]} F2 score:", f2_score)
        print()
