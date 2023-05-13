import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import time
from sklearn.model_selection import StratifiedKFold

def load_data(path):
    data = pd.read_csv(path)
    target = data.iloc[:, 1]
    features =(data.iloc[:, 2:]-data.iloc[:, 2:].min(axis=0))/(data.iloc[:, 2:].max(axis=0)-data.iloc[:, 2:].min(axis=0))
    features.fillna(0,inplace=True)
    return  data,features, target

def split_data(data):
    data = (data - data.min(axis=0)) / (
                data.max(axis=0) - data.min(axis=0))
    mutation = data.iloc[:, 2:14310]
    CNV = data.iloc[:, 14310:39383]
    methylation = data.iloc[:, 39383:-1]
    mutation.fillna(0, inplace=True)
    CNV.fillna(0, inplace=True)
    methylation.fillna(0, inplace=True)
    return mutation, CNV, methylation

def find_best_par(C_max, C_min, num, features, target, multi_class):
    start_time = time.time()
    logistic = LR()
    penalty = ['l1']
    C = np.linspace(C_min, C_max, num)
    solver = ['saga']
    max_iter = [50]
    random_state=[42]
    hyperparameters = dict(C=C, penalty=penalty, solver=solver,random_state=random_state, max_iter=max_iter, multi_class=multi_class)
    # 创建网格搜索对象
    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    randomizedsearchCV = GridSearchCV(logistic, hyperparameters, cv=kflod, n_jobs=-1,scoring='accuracy')
    best_model = randomizedsearchCV.fit(features, target)
    parameters = best_model.best_estimator_.get_params()
    return parameters

def load_data_to222(path,flag):
    data = pd.read_csv(path)
    if flag==1:
        data.loc[data.loc[:, "Subtype_ID"] == 2, "Subtype_ID"] = 3
        data.loc[data.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 3
    elif flag==2:
        data=data.drop(data[data.loc[:,"Subtype_ID"]==4].index)
        data.loc[data.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 2
    else:
        data = data.drop(data[data.loc[:, "Subtype_ID"] == 4].index)
        data = data.drop(data[data.loc[:, "Subtype_ID"] == 3].index)
    target = data.iloc[:, 1]
    features = (data.iloc[:, 2:] - data.iloc[:, 2:].min(axis=0)) / (
                data.iloc[:, 2:].max(axis=0) - data.iloc[:, 2:].min(axis=0))
    features.fillna(0, inplace=True)
    return data, features, target

def load_data_to32(path,flag):
    data = pd.read_csv(path)
    if flag==1:
        data.loc[data.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 2
    elif flag==2:
        data = data.drop(data[data.loc[:, "Subtype_ID"] == 4].index)
        data = data.drop(data[data.loc[:, "Subtype_ID"] == 3].index)
    target = data.iloc[:, 1]
    features = (data.iloc[:, 2:] - data.iloc[:, 2:].min(axis=0)) / (
            data.iloc[:, 2:].max(axis=0) - data.iloc[:, 2:].min(axis=0))
    features.fillna(0, inplace=True)
    return data, features, target

path_train_data = "/home/bks/Challenge/DataSet/SMOTE_train_data.csv"
train_data, train_features_four, train_target = load_data(path_train_data)
train_mutation, train_CNV, train_methylation = split_data(train_data)




# Parameter Adjustment For I-MC
# I-MC(A)
train_C_max_four = 0.5
train_C_min_four =  0.1
train_C_num_four = 50
multi_class = ['ovr']
parameters = find_best_par(train_C_max_four, train_C_min_four, train_C_num_four, train_features_four, train_target,
                                multi_class)

# I-MC(G)
train_C_max_mutation = 0.5
train_C_min_mutation =  0.1
train_C_num_mutation = 50
multi_class = ['ovr']
parameters_mutation = find_best_par(train_C_max_mutation, train_C_min_mutation, train_C_num_mutation, train_mutation,
                                    train_target, multi_class)

# I-MC(C)
train_C_max_CNV =0.5
train_C_min_CNV =   0.1
train_C_num_CNV = 50
multi_class = ['ovr']
parameters_CNV = find_best_par(train_C_max_CNV, train_C_min_CNV, train_C_num_CNV, train_CNV, train_target, multi_class)

# I-MC(M)
train_C_max_methylation = 0.5
train_C_min_methylation =  0.1
train_C_num_methylation = 50
multi_class = ['ovr']
parameters_methylation = find_best_par(train_C_max_methylation, train_C_min_methylation, train_C_num_methylation,
                                       train_methylation,
                                       train_target, multi_class)
print("Parameter Adjustment For I-MC(A)")
print(parameters)
print()
print("Parameter Adjustment For I-MC(G)")
print(parameters_mutation)
print()
print("Parameter Adjustment For I-MC(C)")
print(parameters_CNV)
print("Parameter Adjustment For I-MC(M)")
print(parameters_methylation)
print()




# Parameter adjustment For II-HC
# II-HC(A)
flag=[1,2,3]
train_C_max = 0.8
train_C_min = 0.3
train_C_num = 50
multi_class = ['ovr']
for i in range(len(flag)):
    data1, features1, target1 = load_data_to222(path_train_data, flag[i])
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           features1,
                                           target1, multi_class)
    print("Parameter adjustment For II-HC(A)")
    print(parameters)
    print()

# II-HC(G)
for i in range(len(flag)):
    data1, features1, target1 = load_data_to222(path_train_data, flag[i])
    train_mutation, train_CNV, train_methylation = split_data(data1)
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           train_mutation,
                                           target1, multi_class)
    print("Parameter adjustment For II-HC(A)")
    print(parameters)
    print()

# II-HC(C)
for i in range(len(flag)):
    data1, features1, target1 = load_data_to222(path_train_data, flag[i])
    train_mutation, train_CNV, train_methylation = split_data(data1)
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           train_CNV,
                                           target1, multi_class)
    print("Parameter adjustment For II-HC(A)")
    print(parameters)
    print()

# II-HC(M)
for i in range(len(flag)):
    data1, features1, target1 = load_data_to222(path_train_data, flag[i])
    train_mutation, train_CNV, train_methylation = split_data(data1)
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           train_methylation,
                                           target1, multi_class)
    print("Parameter adjustment For II-HC(A)")
    print(parameters)
    print()








# Parameter adjustment For III-HC
# III-HC(A)
flag=[1,2]
train_C_max =0.8
train_C_min =   0.3
train_C_num = 50
multi_class = ['ovr']
for i in range(len(flag)):
    data1, features1, target1 = load_data_to32(path_train_data, flag[i])
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           features1,
                                           target1, multi_class)
    print("Parameter adjustment For III-HC(A)")
    print(parameters)
    print()

# III-HC(G)
flag=[1,2]
train_C_max =0.8
train_C_min =   0.3
train_C_num = 50
multi_class = ['ovr']
for i in range(len(flag)):
    data1, features1, target1 = load_data_to32(path_train_data, flag[i])
    train_mutation, train_CNV, train_methylation = split_data(data1)
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           train_mutation,
                                           target1, multi_class)
    print("Parameter adjustment For III-HC(G)")
    print(parameters)
    print()

# III-HC(C)
flag=[1,2]
train_C_max =0.8
train_C_min =   0.3
train_C_num = 50
multi_class = ['ovr']
for i in range(len(flag)):
    data1, features1, target1 = load_data_to32(path_train_data, flag[i])
    train_mutation, train_CNV, train_methylation = split_data(data1)
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           train_CNV,
                                           target1, multi_class)
    print("Parameter adjustment For III-HC(C)")
    print(parameters)
    print()

# III-HC(M)
flag=[1,2]
train_C_max =0.8
train_C_min =   0.3
train_C_num = 50
multi_class = ['ovr']
for i in range(len(flag)):
    data1, features1, target1 = load_data_to32(path_train_data, flag[i])
    train_mutation, train_CNV, train_methylation = split_data(data1)
    print("The {}th stage".format(i))
    parameters = find_best_par(train_C_max, train_C_min,
                                           train_C_num,
                                           train_methylation,
                                           target1, multi_class)
    print("Parameter adjustment For III-HC(M)")
    print(parameters)
    print()