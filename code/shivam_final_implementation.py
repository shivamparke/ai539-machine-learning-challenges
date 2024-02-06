__author__ = "Shivam"
__email__ = "shivam@oregonstate.edu"
__date__ = "March 20, 2023"
__assignment__ = "Final Project"

import category_encoders
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import chlng1_count
import chlng1_onehot
import chlng1_ordinal
import chlng2_minmax
import chlng2_robustscaler
import chlng2_standardscaler
import chlng3_imbalance_randover
import chlng3_imbalance_randunder
import chlng4_haversine
import chlng4_replcby_xyz
import fputils


'''
This is the MAIN file that is to be executed to obtain the results for the strategies to resolve each challenge.
'''


def resolve_challenges_independently():
    '''
    This function generates the result of each strategy for each challenge independetly, except when specified explicitly.
    That is, when the best strategy from a challenge is being used for a future challenge, you'll see a print statement.
    :return: None. It prints the results of each strategy for a challenge.
    '''
    chlng1_count.get_count_encoding_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng1_onehot.get_onehot_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng1_ordinal.get_ordinal_encoding_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    print("Using count encoding from here onwards as it was the best encoding scheme.".upper())
    print()

    chlng2_minmax.get_minmax_scaling_results()
    print()
    print("The above is also the result for the 'Do Nothing' strategy for class imbalance.")
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng2_robustscaler.get_robust_scaling_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng2_standardscaler.get_standard_scaling_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng3_imbalance_randover.get_random_oversampling_results()
    print()
    print("The above is also the result for the 'Do Nothing' strategy for transforming location-related information.")
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng3_imbalance_randunder.get_random_undersampling_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng4_haversine.get_haversine_strategy_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()

    chlng4_replcby_xyz.get_replace_with_xyz_results()
    print('----------------------------------------------------------------------------------------------------------')
    print()


def get_the_best_results_overall():
    '''
    Based on the observations so far, this function generates the best possible performance results for the KNN classifier.
    The optimal hyperparameters are used and the best strategy for each challenge is used.
    :return: None.
    '''
    seed = 0
    print("This is the best performance...")
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

    X_old = X
    y_old = y

    # Perform random oversampling
    over_sampler = RandomOverSampler(random_state=seed)
    X_balanced, y_balanced = over_sampler.fit_resample(X, y)
    X = X_balanced
    y = y_balanced
    X_new = X_balanced
    y_new = y_balanced

    # define the KNN classifier object
    classifiers = []
    knn_classifier = KNeighborsClassifier(**fputils.tune_hyperparameters())
    classifiers.append(knn_classifier)
    baseline_classifier = DummyClassifier(strategy="stratified", random_state=seed)
    classifiers.append(baseline_classifier)
    cv_stratified10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for index, classifier in enumerate(classifiers):
        if index == 0:
            X = X_new
            y = y_new
        else:
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


resolve_challenges_independently()

get_the_best_results_overall()

print("---END OF OUTPUT---")
