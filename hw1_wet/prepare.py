from types import prepare_class
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from matplotlib import pylab
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist


# @title prepare.py
def replace_blood_type_with_bolean(data):
    data["001"] = (data["blood_type"].isin(["A+", "A-"])).astype(int)
    data["010"] = (data["blood_type"].isin(["O+", "O-"])).astype(int)
    data["100"] = (data["blood_type"].isin(["AB+", "AB-", "B+", "B-"])).astype(int)


def preprare_data(training_data, new_data):
    # get training set
    trainingSet, testSet = model_selection.train_test_split(training_data, train_size=0.8, test_size=0.2,
                                                            random_state=69)

    # define what to do with cols
    to_delete = ["current_location", "symptoms", "blood_type", "pcr_date"]
    to_standart_norm = ["patient_id", "num_of_siblings", "happiness_score", "household_income", "conversations_per_day",
                        "sugar_levels", "PCR_06", "PCR_07", "PCR_08", "PCR_09"]
    to_min_max_norm = ["happiness_score", "age", "sport_activity", "PCR_01", "PCR_02", "PCR_03", "PCR_04", "PCR_05"]

    # copy to return
    result_data = new_data.copy()

    # special cases
    replace_blood_type_with_bolean(result_data)
    result_data = pd.concat([result_data, result_data['symptoms'].str.get_dummies(sep=';')], axis=1)
    result_data["sex"].replace({"F": 1, "M": -1}, inplace=True)

    replace_blood_type_with_bolean(trainingSet)
    trainingSet = pd.concat([trainingSet, trainingSet['symptoms'].str.get_dummies(sep=';')], axis=1)
    trainingSet["sex"].replace({"F": 1, "M": -1}, inplace=True)
    # delete
    for col in to_delete:
        del result_data[col]
        del trainingSet[col]

    # standart_norm
    standart_scaler = StandardScaler()
    standart_scaler.fit(trainingSet[to_standart_norm])
    result_data[to_standart_norm] = standart_scaler.transform(result_data[to_standart_norm])

    # minmax_norm
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    minmax_scaler.fit(trainingSet[to_min_max_norm])
    result_data[to_min_max_norm] = minmax_scaler.transform(result_data[to_min_max_norm])
    return result_data


trainingsetfornoa, testSet = model_selection.train_test_split(virus_data, train_size=0.8, test_size=0.2,
                                                              random_state=69)

testing = preprare_data(virus_data, trainingsetfornoa)