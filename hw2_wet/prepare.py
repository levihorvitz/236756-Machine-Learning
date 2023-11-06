import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np


def prepare_data(training_data, new_data):
    copy_data = training_data.copy()
    copy_data = processingFeatures(copy_data)
    copy_data = normalization(copy_data)
    new_data = copy_data
    return new_data


def normalization(cpy_data):
    list_minmax = ['age', 'sport_activity', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_07', 'PCR_10', \
                   'x_blood', 'y_blood', 'z_blood']

    list_std = ['weight', 'num_of_siblings', 'happiness_score', 'household_income', 'conversations_per_day', \
                'sugar_levels', 'PCR_06', 'PCR_08', 'PCR_10']

    min_max_scalar = MinMaxScaler((-1, 1))
    standard_scalar = StandardScaler()
    min_max_data = cpy_data[list_minmax].copy()
    standard_data = cpy_data[list_std].copy()
    min_max_scalar.fit(min_max_data)
    standard_scalar.fit(standard_data)

    cpy_data[list_minmax] = min_max_scalar.transform(cpy_data[list_minmax])
    cpy_data[list_std] = standard_scalar.transform(cpy_data[list_std])

    return cpy_data


def processingFeatures(data):
    # turn features to int values
    X_blood = data["blood_type"].isin(['A+', 'A-']).astype(int)
    Y_blood = data["blood_type"].isin(['B+', 'B-', 'AB+', 'AB-']).astype(int)
    Z_blood = data["blood_type"].isin(['O+', 'O-']).astype(int)

    cough = data['symptoms'].isin(data['symptoms'].str.contains('cough')).astype(int)
    low_appetite = data['symptoms'].isin(data['symptoms'].str.contains('low_appetite')).astype(int)
    shortness_of_breath = data['symptoms'].isin(data['symptoms'].str.contains('shortness_of_breath')).astype(int)
    fever = data['symptoms'].isin(data['symptoms'].str.contains('fever')).astype(int)
    sore_throat = data['symptoms'].isin(data['symptoms'].str.contains('sore_throat')).astype(int)

    male = data["sex"].isin(['M']).astype(int)

    # remove features
    data.drop(inplace=True, columns='symptoms')
    data.drop(inplace=True, columns='current_location')
    data.drop(inplace=True, columns='pcr_date')
    data.drop(inplace=True, columns='patient_id')
    data.drop(inplace=True, columns='sex')
    data.drop(inplace=True, columns='blood_type')

    # add features
    data = data.assign(cough=cough, low_appetite=low_appetite, shortness_of_breath=shortness_of_breath, \
                       fever=fever, sore_throat=sore_throat)
    data = data.assign(x_blood=X_blood, y_blood=Y_blood, z_blood=Z_blood)
    data = data.assign(male=male)

    return data
