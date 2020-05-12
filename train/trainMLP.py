
import numpy as np
import pickle
import pandas as pd
import pandas.util.testing as tm
from sklearn.neural_network import MLPClassifier 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__== "__main__":

    # Read The Data
    df = pd.read_csv('data/Data1.csv')
    train, test = data_split(df, 0.2)
    # X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    # X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()


    X_train = train[[ 'Fever',	'Tiredness'	,'Dry-Cough'	,'Difficulty-in-Breathing',	'Sore-Throat','None_Sympton','Pains'	,'Nasal-Congestion'	,'Runny-Nose',	'Age_0-9','Age_10-19'	,'Age_20-24'	,'Age_25-59',	'Age_60+','Severity_Mild',	'Severity_Moderate'	,'Severity_None']].to_numpy()
    X_test = test[[ 'Fever',	'Tiredness'	,'Dry-Cough'	,'Difficulty-in-Breathing',	'Sore-Throat','None_Sympton','Pains'	,'Nasal-Congestion'	,'Runny-Nose',	'Age_0-9','Age_10-19'	,'Age_20-24'	,'Age_25-59',	'Age_60+','Severity_Mild',	'Severity_Moderate'	,'Severity_None']].to_numpy()

    Y_train = train[['Severity_Severe']].to_numpy().reshape(480,)
    Y_test = test[['Severity_Severe']].to_numpy().reshape(119,)
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd',random_state=0).fit(X_train, Y_train) 
    mlp.fit(X_train, Y_train)


    file = open('model/model.pkl','wb')

    pickle.dump(mlp, file)
    file.close()
   

   