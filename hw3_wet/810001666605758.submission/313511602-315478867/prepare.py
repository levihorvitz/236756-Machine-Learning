import pandas as pd
from matplotlib import pylab
from sklearn.model_selection import train_test_split

params = {'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'axes.titlesize': 22,
          'axes.labelsize': 20,
          'legend.fontsize': 18,
          'legend.title_fontsize': 22,
          'figure.titlesize': 24
          }
pylab.rcParams.update(params)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def replace_blood_type_with_bolean(data):
    data["blood_type1"] = (data["blood_type"].isin(["A+", "A-"])).astype(int)
    data["blood_type2"] = (data["blood_type"].isin(["O+", "O-"])).astype(int)
    data["blood_type3"] = (data["blood_type"].isin(["AB+", "AB-", "B+", "B-"])).astype(int)


def prepare_data(raw_data, new_data):
    # get training set
    trainingSet,_ = train_test_split(raw_data, test_size=0.2, random_state=69)
    # define what to do with cols
    to_delete = ["current_location", "symptoms", "blood_type", "pcr_date","patient_id"]
    to_standart_norm = ["happiness_score","num_of_siblings",'weight',   "household_income", "sugar_levels", "PCR_06", "PCR_08",'PCR_10','age', "conversations_per_day"]
    to_min_max_norm = [ "PCR_07",'sport_activity', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05' ,"PCR_09"]

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

    # standard_norm
    standard_scaler = StandardScaler()
    standard_scaler.fit(trainingSet[to_standart_norm])
    result_data[to_standart_norm] = standard_scaler.transform(result_data[to_standart_norm])

    # minmax_norm
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    minmax_scaler.fit(trainingSet[to_min_max_norm])
    result_data[to_min_max_norm] = minmax_scaler.transform(result_data[to_min_max_norm])
    return result_data