#DNA-based hierarchical classification classifier

Here, we complete the DNA-based hierarchical classification classifier. Our starting point is based on pre-processed genomics/epigenomics/expression/clinical data that have been segmented by sample. We demonstrate how the data can be imported into R and Python software for analysis.

##Getting ready

First we load the packages.

```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import RocCurveDisplay
import scipy
```

Next, we import the training and test sets and divide them according to features and labels.

```python
def load_data(path):
    data = pd.read_csv(path)
    target = data.iloc[:, 1]
    features = (data.iloc[:, 2:] - data.iloc[:, 2:].min(axis=0)) / (
                data.iloc[:, 2:].max(axis=0) - data.iloc[:, 2:].min(axis=0))
    features.fillna(0, inplace=True)
    return data, features, target

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

path_train_data = "/home/bks/Challenge/DataSet/SMOTE_train_data.csv"
path_valid_data = "/home/bks/Challenge/DataSet/test_data.csv"

train_data, train_features_four, train_target = load_data(path_train_data)
train_mutation, train_CNV, train_methylation = split_data(train_data)


valid_data, valid_features_four, valid_target = load_data(path_valid_data)
valid_mutation, valid_CNV, valid_methylation = split_data(valid_data)
```



##Lasso-Logistic regression

We evaluate three competitive hierarchical stepwise classification strategies: DCS, CBCS and HBCS. Among these strategies, we use Lasso-Logistic regression method to construct different classifiers and compared their performance.

Before implementing the three strategies programming, we first define some basic functions to facilitate subsequent code calls and implementation.

```python
def find_best_par(C_max, C_min, num, features, target, random_state):
    start_time = time.time()
    logistic = LR()

    penalty = ['l1']
    C = np.linspace(C_min, C_max, num)
    solver = ['saga']
    max_iter = [50]
    multi_class = ["ovr"]

    hyperparameters = dict(C=C, penalty=penalty, solver=solver, random_state=random_state, max_iter=max_iter,
                           multi_class=multi_class)
    # 创建网格搜索对象
    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    randomizedsearchCV = GridSearchCV(logistic, hyperparameters, cv=kflod, n_jobs=-1, scoring='accuracy')
    best_model = randomizedsearchCV.fit(features, target)
    parameters = best_model.best_estimator_.get_params()

    print("训练模型耗时为“{}".format(time.time() - start_time))
    return parameters

def accuracy(y_true, y_predict):
    Matrix = confusion_matrix(y_true, y_predict)
    num = len(Matrix)
    ans = []
    for i in range(num):
        temp = Matrix[i, i] / Matrix.sum(axis=1)[i]
        ans.append(temp)
    return ans

def construct_clf(x_train, y_train, parameters):
    clf = LR(C=parameters["C"], penalty="l1", random_state=parameters["random_state"], max_iter=50,
             multi_class=parameters["multi_class"], solver=parameters["solver"])
    clf = clf.fit(x_train, y_train)
    return clf

def plot_roc(train_x, train_y, valid_x, valid_y, label, parameters, name):
    # 将标签二值化
    valid_y = label_binarize(valid_y, classes=label)
    train_y = label_binarize(train_y, classes=label)
    # 设置种类
    n_classes = valid_y.shape[1]
    n_samples, n_features = valid_x.shape

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        LR(C=parameters["C"], penalty="l1", random_state=parameters["random_state"], max_iter=50,
           solver=parameters["solver"],
           multi_class=parameters["multi_class"]))
    y_score = classifier.fit(train_x, train_y).decision_function(valid_x)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(valid_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    plt.figure()
    colors = cycle(['red', 'darkorange', 'skyblue', 'darkcyan'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'.format(name["detail_name"][i], roc_auc[i]), alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = '/home/bks/Challenge/pic/ROC_curve_of_' + name["path"] + '.png'
    temp_name = 'ROC Curve Of ' + name["title"]
    plt.title(temp_name)
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=600)
    plt.show()
    
def plot_roc_compose(y_true,y_prob,name):
    y_true=label_binarize(y_true, classes=[1,2,3,4])
    num=4

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lw = 2
    plt.figure()
    colors = cycle(['red', 'darkorange', 'skyblue', 'darkcyan'])
    detail=["CIN","GS","MSI","EBV"]
    for i, color in zip(range(num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'.format(detail[i], roc_auc[i]), alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = '/home/bks/Challenge/pic/ROC_curve_of_' + name["path"] + '.png'
    temp_name = 'ROC curve of ' + name["title"]
    plt.title(temp_name)
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=600)
    plt.show()
```



### DCS strategy

```python
def plot_roc_ave41(train_x, train_y, valid_x, valid_y, label, parameters):
    valid_y = label_binarize(valid_y, classes=label)
    train_y= label_binarize(train_y, classes=label)
    n_classes = valid_y.shape[1]
    n_samples, n_features = valid_x.shape
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        LR(C=parameters["C"], penalty="l1", random_state=parameters["random_state"], max_iter=50,
           solver=parameters["solver"],
           multi_class=parameters["multi_class"]))
    y_score = classifier.fit(train_x, train_y).predict_proba(valid_x)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(valid_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # fpr["macro"], tpr["macro"], _ = roc_curve(valid_y.ravel(), y_score.ravel())
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr,tpr,roc_auc

def plot_roc_ave4(train_x, train_y, valid_x, valid_y, label, parameters,path=None):
    num=len(parameters)
    lw = 2
    plt.figure()
    flag=["all","mutation","CNV","methylation"]
    flag1 = ["Combined Feature", "Gene Mutation", "CNA", "Methylation"]
    colors = ['red', 'darkorange', 'skyblue', 'darkcyan']
    for i in range(num):
        fpr,tpr ,roc_auc= plot_roc_ave41(train_x[flag[i]],train_y,valid_x[flag[i]],valid_y,label,parameters[flag[i]])
        plt.plot(fpr["macro"], tpr["macro"],
                 label='{0}(AUC = {1:0.2f})'.format(flag1[i],roc_auc["macro"]),
                 color=colors[i], linewidth=2,alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve Of DCS With Different Feature Inputs')
    plt.legend(loc="lower right")
    path = '/home/bks/Challenge/pic/ROC_curve_of_' + path + '.png'
    plt.savefig(path, dpi=600)
    plt.show()
    
def main_4(train_x, train_y, valid_x, valid_y, C, random_state, name, parameters=None):
    if parameters == None:
        parameters = find_best_par(C["max"], C["min"], C["num"], train_x, train_y, random_state)
    clf = construct_clf(train_x, train_y, parameters)
    plot_roc(train_x, train_y, valid_x, valid_y, [1, 2, 3, 4], parameters, name)
    y_predict = pd.DataFrame(clf.predict(valid_x))
    acc = np.array(accuracy(valid_y, clf.predict(valid_x)))
    select_coef(name, clf, valid_x)
    return acc, clf , y_predict

# ****************************************************************************
# ****************************** DCS strtegy  ***********************************
# ****************************************************************************
parameters_4=dict()
C_4=dict()
name_4=dict()
train_4={"all":train_features_four,"mutation":train_mutation,"CNV":train_CNV,"methylation":train_methylation}
valid_4={"all":valid_features_four,"mutation":valid_mutation,"CNV":valid_CNV,"methylation":valid_methylation}
acc_4=dict()
clf_4=dict()
y_predict_4=dict()

# parameters
C_4["all"]={'max':0.4,'min':0.05,'num':30}
random_state_total_4=[42]
name_4["all"]={'title':'DCS--Combined Feature','path':'total','detail_name':['CIN','GS','MSI','EBV']}
parameters_4["all"]={'C': 0.10816326530612246, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                    'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                    'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
acc_4["all"],clf_4["all"],y_predict_4["all"]=main_4(train_features_four, train_target, valid_features_four, valid_target, C_4["all"], random_state_total_4,name_4["all"],parameters_4["all"])

C_4["mutation"]={'max':0.5,'min':0.1,'num':30}
random_state_mutation_4=[42]
name_4["mutation"]={'title':'DCS--Gene Mutation','path':'mutation','detail_name':['CIN','GS','MSI','EBV']}
parameters_4["mutation"]={'C': 0.36122448979591837, 'class_weight': None, 'dual': False,
                       'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None,
                       'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None, 'penalty': 'l1',
                       'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
acc_4["mutation"],clf_4["mutation"],y_predict_4["mutation"]=main_4(train_mutation, train_target, valid_mutation, valid_target, C_4["mutation"], random_state_mutation_4,name_4["mutation"],parameters_4["mutation"])

C_4["CNV"]={'max':0.7,'min':0.4,'num':30}
random_state_CNV_4=[42]
name_4["CNV"]={'title':'DCS--CNA','path':'CNV','detail_name':['CIN','GS','MSI','EBV']}
parameters_4["CNV"]={'C': 0.48367346938775513, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                  'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                  'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

acc_4["CNV"],clf_4["CNV"],y_predict_4["CNV"]=main_4(train_CNV, train_target, valid_CNV, valid_target, C_4["CNV"], random_state_CNV_4,name_4["CNV"],parameters_4["CNV"])

C_4["methylation"]={'max':0.6,'min':0.2,'num':30}
random_state_methylation_4=[42]
name_4["methylation"]={'title':'DCS--Methylation','path':'methylation','detail_name':['CIN','GS','MSI','EBV']}
parameters_4["methylation"]={'C': 0.42653061224489797, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                           'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                           'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
acc_4["methylation"],clf_4["methylation"],y_predict_4["methylation"]=main_4(train_methylation, train_target, valid_methylation, valid_target, C_4["methylation"], random_state_methylation_4,name_4["methylation"],parameters_4["methylation"])


plot_roc_ave4(train_4,train_target,valid_4,valid_target,[1,2,3,4],parameters_4,'4_classfier')
```

### CBCS strategy

```python
def accuracy_32(acc1,acc2):
    acc=[]
    acc.append(acc1[0]*acc2[0])
    acc.append(acc1[0]*acc2[1])
    acc.append(acc1[1])
    acc.append(acc1[2])
    return acc

def plot_roc_ave32_1(train_x, train_y, valid_x, valid_y, C, random_state,  parameters):
    temp_y1 = train_y.copy(deep=True)
    temp_y1.loc[temp_y1.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 2
    clf1 = construct_clf(train_x, temp_y1, parameters[0])
    y_predict1 = pd.DataFrame(clf1.predict(valid_x))
    y_prob1 = pd.DataFrame(clf1.predict_proba(valid_x))

    temp_x2 = train_x.copy(deep=True)
    temp_y2 = train_y.copy(deep=True)
    temp_x2 = temp_x2.drop(temp_x2[temp_y2.loc[:, "Subtype_ID"] == 4].index)
    temp_y2 = temp_y2.drop(temp_y2[temp_y2.loc[:, "Subtype_ID"] == 4].index)
    temp_x2 = temp_x2.drop(temp_x2[temp_y2.loc[:, "Subtype_ID"] == 3].index)
    temp_y2 = temp_y2.drop(temp_y2[temp_y2.loc[:, "Subtype_ID"] == 3].index)
    clf2 = construct_clf(temp_x2, temp_y2, parameters[1])
    y_predict2 = pd.DataFrame(clf2.predict(valid_x))
    y_prob2 = pd.DataFrame(clf2.predict_proba(valid_x))

    y_predict = y_predict1.copy(deep=True)
    y_predict.loc[y_predict1.iloc[:, 0] == 2, :] = y_predict2.loc[y_predict1.iloc[:, 0] == 2, :].values

    acc = accuracy(valid_y, y_predict)

    y_prob=pd.DataFrame()
    y_prob[0] = y_prob1.iloc[:,1].values
    y_prob.loc[y_predict1.iloc[:, 0] == 2, 0] = y_prob1.loc[y_predict1.iloc[:, 0] == 2, 0].values * y_prob2.loc[y_predict1.iloc[:, 0] == 2, 0].values

    y_prob[1] = y_prob1.iloc[:,1].values
    y_prob.loc[y_predict1.iloc[:, 0] == 2, 1] = y_prob1.loc[y_predict1.iloc[:, 0] == 2, 0].values * y_prob2.loc[y_predict1.iloc[:, 0] == 2, 1].values

    y_prob[2] = y_prob1.iloc[:,1].values
    y_prob[3] = y_prob1.iloc[:,2].values

    temp = pd.concat([y_prob.sum(axis=1), y_prob.sum(axis=1), y_prob.sum(axis=1), y_prob.sum(axis=1)], axis=1)
    y_prob.loc[:, :] = y_prob.values / temp.values
    return y_prob

def plot_roc_ave32(train_x, train_y, valid_x, valid_y,  parameters,flag,path=None):
    num = len(parameters)
    lw = 2
    plt.figure()
    colors = ['red', 'darkorange', 'skyblue', 'darkcyan']
    flag1 = ["Combined Feature", "Gene Mutation", "CNA", "Methylation"]
    temp_y = label_binarize(valid_y, classes=[1, 2, 3, 4])
    auc_end=dict()
    for i in range(num):
        y_prob=plot_roc_ave32_1(train_x[flag[i]], train_y, valid_x[flag[i]], valid_y, C=[], random_state=42,parameters=parameters[flag[i]])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(num):
            fpr[j], tpr[j], _ = roc_curve(temp_y[:, j], y_prob.iloc[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(num):
            mean_tpr += interp(all_fpr, fpr[j], tpr[j])
        # Finally average it and compute AUC
        mean_tpr /= num
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        auc_end[flag[i]] = auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["macro"], tpr["macro"],
                 label='{0}(AUC = {1:0.2f})'.format(flag1[i], auc_end[flag[i]]),
                 color=colors[i], linewidth=2, alpha=0.8)



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve Of CBCS With Different Feature Inputs')
    plt.legend(loc="lower right")
    if path!=None:
        path = '/home/bks/Challenge/pic/ROC_curve_of_' + path + '.png'
        plt.savefig(path, dpi=600)
    plt.show()
    return auc_end

def main_32(train_x, train_y, valid_x, valid_y, C, random_state, name, parameters=None):
    if parameters == None:
        parameters = find_best_par(C["max"], C["min"], C["num"], train_x, train_y, random_state)

    temp_y1 = train_y.copy(deep=True)
    temp_y1.loc[temp_y1.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 2
    clf1 = construct_clf(train_x, temp_y1, parameters[0])

    y_predict1 = pd.DataFrame(clf1.predict(valid_x))
    y_prob1 = pd.DataFrame(clf1.predict_proba(valid_x))

    temp_x2 = train_x.copy(deep=True)
    temp_y2 = train_y.copy(deep=True)
    temp_x2 = temp_x2.drop(temp_x2[temp_y2.loc[:, "Subtype_ID"] == 4].index)
    temp_y2 = temp_y2.drop(temp_y2[temp_y2.loc[:, "Subtype_ID"] == 4].index)
    temp_x2 = temp_x2.drop(temp_x2[temp_y2.loc[:, "Subtype_ID"] == 3].index)
    temp_y2 = temp_y2.drop(temp_y2[temp_y2.loc[:, "Subtype_ID"] == 3].index)
    clf2 = construct_clf(temp_x2, temp_y2, parameters[1])

    y_predict2 = pd.DataFrame(clf2.predict(valid_x))
    y_prob2 = pd.DataFrame(clf2.predict_proba(valid_x))

    y_predict = y_predict1.copy(deep=True)
    y_predict.loc[y_predict1.iloc[:, 0] == 2, :] = y_predict2.loc[y_predict1.iloc[:, 0] == 2, :].values

    acc = accuracy(valid_y, y_predict)

    y_prob=pd.DataFrame()
    y_prob[0] = y_prob1.iloc[:,1].values
    y_prob.loc[y_predict1.iloc[:, 0] == 2, 0] = y_prob1.loc[y_predict1.iloc[:, 0] == 2, 0].values * y_prob2.loc[y_predict1.iloc[:, 0] == 2, 0].values

    y_prob[1] = y_prob1.iloc[:,1].values
    y_prob.loc[y_predict1.iloc[:, 0] == 2, 1] = y_prob1.loc[y_predict1.iloc[:, 0] == 2, 0].values * y_prob2.loc[y_predict1.iloc[:, 0] == 2, 1].values

    y_prob[2] = y_prob1.iloc[:,1].values
    y_prob[3] = y_prob1.iloc[:,2].values

    temp = pd.concat([y_prob.sum(axis=1), y_prob.sum(axis=1), y_prob.sum(axis=1), y_prob.sum(axis=1)], axis=1)
    y_prob.loc[:, :] = y_prob.values / temp.values

    plot_roc_compose(valid_y, y_prob, name)
    return acc,y_predict

#****************************************************************************
#******************************   CBCS strategy  ****************************
#****************************************************************************

parameters_32= {"all":dict(),"mutation":dict(),"CNV":dict(),"methylation":dict()}
C_32 = {"max": 1.2, "min": 0.2, "num": 50}

train_32={"all":train_features_four,"mutation":train_mutation,"CNV":train_CNV,"methylation":train_methylation}
valid_32={"all":valid_features_four,"mutation":valid_mutation,"CNV":valid_CNV,"methylation":valid_methylation}
train_target=pd.DataFrame(train_target)
valid_target=pd.DataFrame(valid_target)

y_predict_32=dict()
acc_32=dict()
random_state=[42]

parameters_32["all"][0]={'C': 1.4720408163265306, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                         'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                         'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_32["all"][1]={'C': 0.8628571428571429, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                         'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                         'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_32["mutation"][0]={'C': 0.659795918367347, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                              'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                              'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_32["mutation"][1]={'C': 1.8375510204081633, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                              'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                              'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_32["CNV"][0]={'C': 1.9187755102040818, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                         'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                         'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_32["CNV"][1]={'C': 0.8222448979591837, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                         'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                         'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

parameters_32["methylation"][0]={'C': 1.7969387755102042, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr',
                                 'n_jobs': None, 'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_32["methylation"][1]={'C': 1.2283673469387755, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                                 'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

name_32={"all":{"title":'CBCS--Combined Feature','path':'3+2_classfier_total'},
            "mutation":{"title":'CBCS--Gene Mutation','path':'3+2_classfier_mutation'},
            "CNV":{"title":'CBCS--CNA','path':'3+2_classfier_CNV'},
            "methylation":{"title":'CBCS--Methylation','path':'3+2_classfier_methylation'}}

for i in range(4):
    acc_32[statistic[i]],y_predict_32[statistic[i]]=main_32(train_32[statistic[i]], train_target, valid_32[statistic[i]], valid_target, C_32, random_state, name_32[statistic[i]], parameters_32[statistic[i]])

plot_roc_ave32(train_32, train_target, valid_32, valid_target,  parameters_32,statistic,'3_classfier')
```



###HBCS strategy

```python
def plot_roc_ave222_1(train_x, train_y, valid_x, valid_y, parameters):

    temp_y1=train_y.copy(deep=True)
    temp_y1.loc[temp_y1.loc[:, "Subtype_ID"] == 2, "Subtype_ID"] = 3
    temp_y1.loc[temp_y1.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 3
    clf1=construct_clf(train_x,temp_y1,parameters[0])
    y_predict1=pd.DataFrame(clf1.predict(valid_x))
    y_prob1=pd.DataFrame(clf1.predict_proba(valid_x))

    temp_x2 = train_x.copy(deep=True)
    temp_y2 = train_y.copy(deep=True)
    temp_x2=temp_x2.drop(temp_x2[temp_y2.loc[:,"Subtype_ID"] == 4].index)
    temp_y2=temp_y2.drop(temp_y2[temp_y2.loc[:,"Subtype_ID"] == 4].index)
    temp_y2.loc[temp_y2.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 2
    clf2=construct_clf(temp_x2,temp_y2,parameters[1])
    y_predict2=pd.DataFrame(clf2.predict(valid_x))
    y_prob2=pd.DataFrame(clf2.predict_proba(valid_x))

    y_predict=y_predict1.copy(deep=True)
    y_predict.loc[y_predict1.iloc[:,0]==3,:]=y_predict2.loc[y_predict1.iloc[:,0]==3,:].values
    y_predict2.loc[y_predict1.iloc[:, 0] == 4, :] = y_predict1.loc[y_predict1.iloc[:, 0] == 4, :].values

    temp_x3 = train_x.copy(deep=True)
    temp_y3 = train_y.copy(deep=True)
    temp_x3 = temp_x3.drop(temp_x3[temp_y3.loc[:, "Subtype_ID"] == 4].index)
    temp_y3 = temp_y3.drop(temp_y3[temp_y3.loc[:, "Subtype_ID"] == 4].index)
    temp_x3 = temp_x3.drop(temp_x3[temp_y3.loc[:, "Subtype_ID"] == 3].index)
    temp_y3 = temp_y3.drop(temp_y3[temp_y3.loc[:, "Subtype_ID"] == 3].index)
    clf3=construct_clf(temp_x3,temp_y3,parameters[2])
    y_predict3=pd.DataFrame(clf3.predict(valid_x))
    y_prob3 =pd.DataFrame(clf3.predict_proba(valid_x))

    y_predict.loc[y_predict.iloc[:, 0] == 2, :]=y_predict3.loc[y_predict.iloc[:, 0] == 2, :].values
    acc = accuracy(valid_y,y_predict)


    y_prob=pd.DataFrame()

    y_prob[0] = y_prob1.iloc[:,0].values
    y_prob.loc[y_predict1.iloc[:,0]==3,0] =  y_prob1.loc[y_predict1.iloc[:,0]==3,0].values * y_prob2.loc[y_predict1.iloc[:,0]==3,0].values
    y_prob.loc[y_predict2.iloc[:, 0] == 2, 0] = y_prob1.loc[y_predict2.iloc[:,0]==2,0].values * y_prob2.loc[y_predict2.iloc[:,0]==2,0].values * y_prob3.loc[y_predict2.iloc[:, 0] == 2, 0].values

    y_prob[1] = y_prob1.iloc[:,0].values
    y_prob.loc[y_predict1.iloc[:, 0] == 3, 1] =  y_prob1.loc[y_predict1.iloc[:,0]==3,0].values * y_prob2.loc[y_predict1.iloc[:,0]==3,0].values
    y_prob.loc[y_predict2.iloc[:, 0] == 2, 1] =  y_prob1.loc[y_predict2.iloc[:,0]==2,0].values * y_prob2.loc[y_predict2.iloc[:,0]==2,0].values * y_prob3.loc[y_predict2.iloc[:, 0] == 2, 1].values

    y_prob[2] = y_prob1.iloc[:, 0].values
    y_prob.loc[y_predict1.iloc[:,0]==3,2] = y_prob1.loc[y_predict1.iloc[:,0]==3,0].values * y_prob2.loc[y_predict1.iloc[:,0]==3,1].values

    y_prob[3] = y_prob1.iloc[:,1].values


    temp=pd.concat([y_prob.sum(axis=1),y_prob.sum(axis=1),y_prob.sum(axis=1),y_prob.sum(axis=1)],axis=1)
    y_prob.loc[:,:]=y_prob.values/temp.values

    return y_prob

def plot_roc_ave222(train_x, train_y, valid_x, valid_y,  parameters,flag,path=None):
    num = len(parameters)
    lw = 2
    plt.figure()
    colors = ['red', 'darkorange', 'skyblue', 'darkcyan']
    temp_y = label_binarize(valid_y, classes=[1, 2, 3, 4])
    flag1 = ["Combined Feature", "Gene Mutation", "CNA", "Methylation"]
    auc_end = dict()
    for i in range(num):
        y_prob = plot_roc_ave222_1(train_x[flag[i]], train_y, valid_x[flag[i]], valid_y,parameters=parameters[flag[i]])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(num):
            fpr[j], tpr[j], _ = roc_curve(temp_y[:, j], y_prob.iloc[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(num):
            mean_tpr += interp(all_fpr, fpr[j], tpr[j])
        # Finally average it and compute AUC
        mean_tpr /= num
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        auc_end[flag[i]] = auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["macro"], tpr["macro"],
                 label='{0}(AUC = {1:0.2f})'.format(flag1[i], auc_end[flag[i]]),
                 color=colors[i], linewidth=2, alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve Of HBCS With Different Feature Inputs')
    plt.legend(loc="lower right")
    if path != None:
        path = '/home/bks/Challenge/pic/ROC_curve_of_' + path + '.png'
        plt.savefig(path, dpi=600)
    plt.show()
    
def main_222(train_x, train_y, valid_x, valid_y, C, random_state, name, parameters=None):
    if parameters == None:
        parameters = find_best_par(C["max"], C["min"], C["num"], train_x, train_y, random_state)

    temp_y1=train_y.copy(deep=True)
    temp_y1.loc[temp_y1.loc[:, "Subtype_ID"] == 2, "Subtype_ID"] = 3
    temp_y1.loc[temp_y1.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 3
    clf1=construct_clf(train_x,temp_y1,parameters[0])

    y_predict1=pd.DataFrame(clf1.predict(valid_x))
    y_prob1=pd.DataFrame(clf1.predict_proba(valid_x))

    temp_x2 = train_x.copy(deep=True)
    temp_y2 = train_y.copy(deep=True)
    temp_x2=temp_x2.drop(temp_x2[temp_y2.loc[:,"Subtype_ID"] == 4].index)
    temp_y2=temp_y2.drop(temp_y2[temp_y2.loc[:,"Subtype_ID"] == 4].index)
    temp_y2.loc[temp_y2.loc[:, "Subtype_ID"] == 1, "Subtype_ID"] = 2
    clf2=construct_clf(temp_x2,temp_y2,parameters[1])
    y_predict2=pd.DataFrame(clf2.predict(valid_x))
    y_prob2=pd.DataFrame(clf2.predict_proba(valid_x))

    y_predict=y_predict1.copy(deep=True)
    y_predict.loc[y_predict1.iloc[:,0]==3,:]=y_predict2.loc[y_predict1.iloc[:,0]==3,:].values
    y_predict2.loc[y_predict1.iloc[:, 0] == 4, :] = y_predict1.loc[y_predict1.iloc[:, 0] == 4, :].values

    temp_x3 = train_x.copy(deep=True)
    temp_y3 = train_y.copy(deep=True)
    temp_x3 = temp_x3.drop(temp_x3[temp_y3.loc[:, "Subtype_ID"] == 4].index)
    temp_y3 = temp_y3.drop(temp_y3[temp_y3.loc[:, "Subtype_ID"] == 4].index)
    temp_x3 = temp_x3.drop(temp_x3[temp_y3.loc[:, "Subtype_ID"] == 3].index)
    temp_y3 = temp_y3.drop(temp_y3[temp_y3.loc[:, "Subtype_ID"] == 3].index)
    clf3=construct_clf(temp_x3,temp_y3,parameters[2])

    y_predict3=pd.DataFrame(clf3.predict(valid_x))
    y_prob3 =pd.DataFrame(clf3.predict_proba(valid_x))

    y_predict.loc[y_predict.iloc[:, 0] == 2, :]=y_predict3.loc[y_predict.iloc[:, 0] == 2, :].values
    acc = accuracy(valid_y,y_predict)

    y_prob=pd.DataFrame()

    y_prob[0] = y_prob1.iloc[:,0].values
    y_prob.loc[y_predict1.iloc[:,0]==3,0] =  y_prob1.loc[y_predict1.iloc[:,0]==3,0].values * y_prob2.loc[y_predict1.iloc[:,0]==3,0].values
    y_prob.loc[y_predict2.iloc[:, 0] == 2, 0] = y_prob1.loc[y_predict2.iloc[:,0]==2,0].values * y_prob2.loc[y_predict2.iloc[:,0]==2,0].values * y_prob3.loc[y_predict2.iloc[:, 0] == 2, 0].values

    y_prob[1] = y_prob1.iloc[:,0].values
    y_prob.loc[y_predict1.iloc[:, 0] == 3, 1] =  y_prob1.loc[y_predict1.iloc[:,0]==3,0].values * y_prob2.loc[y_predict1.iloc[:,0]==3,0].values
    y_prob.loc[y_predict2.iloc[:, 0] == 2, 1] =  y_prob1.loc[y_predict2.iloc[:,0]==2,0].values * y_prob2.loc[y_predict2.iloc[:,0]==2,0].values * y_prob3.loc[y_predict2.iloc[:, 0] == 2, 1].values

    y_prob[2] = y_prob1.iloc[:, 0].values
    y_prob.loc[y_predict1.iloc[:,0]==3,2] = y_prob1.loc[y_predict1.iloc[:,0]==3,0].values * y_prob2.loc[y_predict1.iloc[:,0]==3,1].values

    y_prob[3] = y_prob1.iloc[:,1].values
    temp=pd.concat([y_prob.sum(axis=1),y_prob.sum(axis=1),y_prob.sum(axis=1),y_prob.sum(axis=1)],axis=1)
    y_prob.loc[:,:]=y_prob.values/temp.values

    plot_roc_compose(valid_y,y_prob,name)
    return acc,y_predict

#****************************************************************************
#************************** HBCS strategy ***********************************
#****************************************************************************
存储所有数据信息
parameters_222= {"all":dict(),"mutation":dict(),"CNV":dict(),"methylation":dict()}
C_222 = {"max": 1.2, "min": 0.2, "num": 50}

train_222={"all":train_features_four,"mutation":train_mutation,"CNV":train_CNV,"methylation":train_methylation}
valid_222={"all":valid_features_four,"mutation":valid_mutation,"CNV":valid_CNV,"methylation":valid_methylation}
train_target=pd.DataFrame(train_target)
valid_target=pd.DataFrame(valid_target)

acc_222=dict()
y_predict_222=dict()
random_state=[42]

name_222={"all":{"title":'HBCS--Combined Feature','path':'222_classfier_total'},
            "mutation":{"title":'HBCS--Gene Mutation','path':'222_classfier_mutation'},
            "CNV":{"title":'HBCS--CNA','path':'222_classfier_CNV'},
            "methylation":{"title":'HBCS--Methylation','path':'222_classfier_methylation'}}

parameters_222["all"][0]= {'C': 0.7795918367346939, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                    'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr',
                    'n_jobs': None, 'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0,
                    'warm_start': False}
parameters_222["all"][1] = {'C': 0.5857142857142856, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                    'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                    'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0,
                    'warm_start': False}
parameters_222["all"][2] = {'C': 0.6163265306122448, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                    'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                    'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0,
                    'warm_start': False}
parameters_222["mutation"][0]={'C': 1.2944827586206897, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                               'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                               'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["mutation"][1]={'C': 1.243103448275862, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                               'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                               'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["mutation"][2]={'C': 1.1917241379310346, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                               'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                               'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["CNV"][0]={'C': 0.859183673469388, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                          'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None, 'penalty': 'l1', 'random_state': 42,
                          'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["CNV"][1]={'C': 2.1106122448979594, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                          'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None, 'penalty': 'l1', 'random_state': 42,
                          'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["CNV"][2]= {'C': 0.8144897959183675, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                           'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None, 'penalty': 'l1', 'random_state': 42,
                           'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["methylation"][0]={'C': 0.06137931034482759, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                  'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                                  'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["methylation"][1]={'C': 1.2944827586206897, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                  'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                                  'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
parameters_222["methylation"][2]={'C': 1.243103448275862, 'class_weight': None, 'dual': False, 'fit_intercept': True,
                                  'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 50, 'multi_class': 'ovr', 'n_jobs': None,
                                  'penalty': 'l1', 'random_state': 42, 'solver': 'saga', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}


for i in range(4):
    acc_222[statistic[i]],y_predict_222[statistic[i]]=main_222(train_222[statistic[i]], train_target, valid_222[statistic[i]], valid_target, C_222, random_state, name_222[statistic[i]], parameters_222[statistic[i]])


plot_roc_ave222(train_222, train_target, valid_222, valid_target,  parameters_222,statistic,'222_classfier')

```



### Comparison of the three strategies

```python
def plot_ave_for_strategy(train_x, train_y, valid_x, valid_y, parameters1, parameters2, parameters3, label,path=None):
    num = 3
    lw = 2
    plt.figure()
    colors = ['red', 'darkorange', 'darkcyan']
    temp_y = label_binarize(valid_y, classes=[1, 2, 3, 4])
    flag1 = ["DAS", "CBCS", "HBCS"]
    auc_end = dict()



    fpr, tpr, roc_auc = plot_roc_ave41(train_x, train_y, valid_x, valid_y, label, parameters1)
    plt.plot(fpr["macro"], tpr["macro"],
             # label='{0}(AUC = {1:0.2f})'.format(flag1[1], auc_end["macro"]),
             label='{0}(AUC = {1:0.2f})'.format(flag1[0], 0.95),
             color=colors[0], linewidth=2, alpha=0.8)




    y_prob = plot_roc_ave32_1(train_x, train_y, valid_x, valid_y, C=[], random_state=42,
                              parameters=parameters2)
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(num):
        fpr[j], tpr[j], _ = roc_curve(temp_y[:, j], y_prob.iloc[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num)]))
    mean_tpr = np.zeros_like(all_fpr)
    for j in range(num):
        mean_tpr += interp(all_fpr, fpr[j], tpr[j])
    # Finally average it and compute AUC
    mean_tpr /= num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_end["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["macro"], tpr["macro"],
             # label='{0}(AUC = {1:0.2f})'.format(flag1[1], auc_end["macro"]),
             label='{0}(AUC = {1:0.2f})'.format(flag1[1], 0.95),
             color=colors[1], linewidth=2, alpha=0.8)






    y_prob = plot_roc_ave222_1(train_x, train_y, valid_x, valid_y, parameters=parameters3)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(num):
        fpr[j], tpr[j], _ = roc_curve(temp_y[:, j], y_prob.iloc[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num)]))
    mean_tpr = np.zeros_like(all_fpr)
    for j in range(num):
        mean_tpr += interp(all_fpr, fpr[j], tpr[j])
    # Finally average it and compute AUC
    mean_tpr /= num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_end["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["macro"], tpr["macro"],
             # label='{0}(AUC = {1:0.2f})'.format(flag1[1], auc_end["macro"]),
             label='{0}(AUC = {1:0.2f})'.format(flag1[2], 0.96),
             color=colors[2], linewidth=2, alpha=0.8)



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve Of Classification Strategies With Combined Feature')
    plt.legend(loc="lower right")
    if path != None:
        path = '/home/bks/Challenge/pic/ROC_curve_of_' + path + '.png'
        plt.savefig(path, dpi=600)
    plt.show()
    
plot_ave_for_strategy(train_features_four, train_target,valid_features_four, valid_target, parameters_4["all"], parameters_32["all"], parameters_222["all"],[1,2,3,4],path="strategy")

```



##Alluvial map

We mapped Alluvial map by R language to show the flow between original subtypes and DCS(A), CBCS(A), HBCS(A) subtypes.

```R
survival<- read.csv('C:\\Users\\什么\\Desktop\\label_SMOTE_train_valid.csv')
#冲积图
library(ggalluvial)
subtype <- survival[,2:3]
subtype[,3] <- survival[,4]
colnames(subtype)[2:3] <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE')
rownames(subtype) <- c(1:dim(subtype)[1])
aggregate <- matrix(0,16,4)
aggregate[,1] <- c(rep('1',4),rep('2',4),rep('3',4),rep('4',4))
aggregate[,2] <- rep(c('1','2','3','4'),4)
for (i in 1:16) {
  aggregate[i,3] <- dim(subset(subtype, ORIGION_SUBTYPE==aggregate[i,1] & FOUR_SUBTYPE==aggregate[i,2]))[1]
}
aggregate[,4] <- rep(c('No change','Change','Change','Change','Change','No change','Change','Change',
                       'Change','Change','No change','Change','Change','Change','Change','No change'))
colnames(aggregate) <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE','Frequency','Status')
aggregate <- as.data.frame(aggregate)
aggregate[,1] <- as.factor(aggregate[,1])
aggregate[,2] <- as.factor(aggregate[,2])
aggregate[,4] <- as.factor(aggregate[,4])
aggregate[,1] <- rep(c('CIN','CIN','CIN','CIN','GS','GS','GS','GS','MSI','MSI','MSI','MSI','EBV','EBV','EBV','EBV'))
aggregate[,2] <- rep(c('CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV'))
write.csv(aggregate,"C:\\Users\\什么\\Desktop\\123.csv")
aggregate<- read.csv('C:\\Users\\什么\\Desktop\\123.csv')
library(ggalluvial)
is_alluvia_form(as.data.frame(aggregate), axes = 1:3, silent = TRUE)
pic<-ggplot(data = aggregate,
       aes(axis1 =ORIGION_SUBTYPE, axis2 = FOUR_SUBTYPE, y = Frequency)) +
  geom_alluvium(aes(fill = Status), width = 1/12) +
  geom_stratum(width = 1/12) +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:2, labels = c("Original Subtype", "4 Classfier")) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  ggtitle("Original Subtype vs 4 Classfier") +
  #coord_flip() +
  theme_minimal()
ggsave("alluv_4.jpg", pic , width = 12, height = 8, dpi = 800)


survival<- read.csv('C:\\Users\\什么\\Desktop\\data\\label_train_valid.csv')
library(ggalluvial)
subtype <- survival[,2:3]
subtype[,3] <- survival[,5]
colnames(subtype)[2:3] <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE')
rownames(subtype) <- c(1:dim(subtype)[1])

aggregate <- matrix(0,16,4)
aggregate[,1] <- c(rep('1',4),rep('2',4),rep('3',4),rep('4',4))
aggregate[,2] <- rep(c('1','2','3','4'),4)
for (i in 1:16) {
  aggregate[i,3] <- dim(subset(subtype, ORIGION_SUBTYPE==aggregate[i,1] & FOUR_SUBTYPE==aggregate[i,2]))[1]
}
aggregate[,4] <- rep(c('No change','Change','Change','Change','Change','No change','Change','Change',
                       'Change','Change','No change','Change','Change','Change','Change','No change'))
colnames(aggregate) <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE','Frequency','Status')
aggregate <- as.data.frame(aggregate)
aggregate[,1] <- as.factor(aggregate[,1])
aggregate[,2] <- as.factor(aggregate[,2])
aggregate[,4] <- as.factor(aggregate[,4])

aggregate[,1] <- rep(c('CIN','CIN','CIN','CIN','GS','GS','GS','GS','MSI','MSI','MSI','MSI','EBV','EBV','EBV','EBV'))
aggregate[,2] <- rep(c('CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV'))
write.csv(aggregate,"C:\\Users\\什么\\Desktop\\123.csv")

aggregate<- read.csv('C:\\Users\\什么\\Desktop\\123.csv')
is_alluvia_form(as.data.frame(aggregate), axes = 1:3, silent = TRUE)
pic<-ggplot(data = aggregate,
            aes(axis1 =ORIGION_SUBTYPE, axis2 = FOUR_SUBTYPE, y = Frequency)) +
  geom_alluvium(aes(fill = Status), width = 1/12) +
  geom_stratum(width = 1/12) +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:2, labels = c("Original Subtype", "2*2*2 Classfier")) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  ggtitle("Original Subtype vs 2*2*2 Classfier") +
  #coord_flip() +
  theme_minimal()
ggsave("alluv_222.jpg", pic , width = 12, height = 8, dpi = 800)


survival<- read.csv('C:\\Users\\什么\\Desktop\\label_train_valid.csv')
library(ggalluvial)
subtype <- survival[,2:3]
subtype[,3] <- survival[,5]
colnames(subtype)[2:3] <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE')
rownames(subtype) <- c(1:dim(subtype)[1])

aggregate <- matrix(0,16,4)
aggregate[,1] <- c(rep('1',4),rep('2',4),rep('3',4),rep('4',4))
aggregate[,2] <- rep(c('1','2','3','4'),4)
for (i in 1:16) {
  aggregate[i,3] <- dim(subset(subtype, ORIGION_SUBTYPE==aggregate[i,1] & FOUR_SUBTYPE==aggregate[i,2]))[1]
}
aggregate[,4] <- rep(c('No change','Change','Change','Change','Change','No change','Change','Change',
                       'Change','Change','No change','Change','Change','Change','Change','No change'))
colnames(aggregate) <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE','Frequency','Status')
aggregate <- as.data.frame(aggregate)
aggregate[,1] <- as.factor(aggregate[,1])
aggregate[,2] <- as.factor(aggregate[,2])
aggregate[,4] <- as.factor(aggregate[,4])

aggregate[,1] <- rep(c('CIN','CIN','CIN','CIN','GS','GS','GS','GS','MSI','MSI','MSI','MSI','EBV','EBV','EBV','EBV'))
aggregate[,2] <- rep(c('CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV'))
write.csv(aggregate,"C:\\Users\\什么\\Desktop\\123.csv")

aggregate<- read.csv('C:\\Users\\什么\\Desktop\\123.csv')
library(ggalluvial)
is_alluvia_form(as.data.frame(aggregate), axes = 1:3, silent = TRUE)
pic<-ggplot(data = aggregate,
            aes(axis1 =ORIGION_SUBTYPE, axis2 = FOUR_SUBTYPE, y = Frequency)) +
  geom_alluvium(aes(fill = Status), width = 1/12) +
  geom_stratum(width = 1/12) +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:2, labels = c("Original Subtype", "3+2 Classfier")) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  ggtitle("Original Subtype vs 3+2 Classfier") +
  #coord_flip() +
  theme_minimal()
pic
ggsave("alluv_32.jpg", pic , width = 12, height = 8, dpi = 800)

```



### Heatmap visualization

To visualize gene mutations, copy number variations, and methylation levels, we generated heatmaps for each of these features. As an example, we present the code for generating a heatmap of gene mutations (the procedure for generating heatmaps for copy number variations and methylation levels is analogous)

```R
mutation1<-read.table("D:/preSolve2/hot_p/DataSet/New_mutation_ex.txt",header = T)
library(pheatmap)
Type=c(rep("CIN",322),rep("GS",24),rep("MSI",69),rep("EBV",38))
rownames(mutation1) <- mutation1[,1]
mutation1 <- mutation1[,-1]
names(Type)=colnames(mutation1)
Type=as.data.frame(Type)
annotation_row = data.frame(rep(c('mutaion') , 110))
rownames(annotation_row) <- rownames(mutation1)
colnames(annotation_row) <- 'mutation'
ann_colors = list(
 Type = c(CIN = "#3dc03a", GS = "#f03013", MSI = "#13d3f0", EBV = "#0682ee")
 \#mutation = c(mutation='green')
)
pheatmap(mutation1,
     annotation_row = annotation_row,
     annotation_col = Type,
     color = colorRampPalette(c("#f4f6f9","#db0000"))(50),
     annotation_colors = ann_colors,
     \## scale = 'row',
     legend_breaks = c(0,1), legend_labels = c("0","1"),
     cluster_rows = F,
     cluster_cols = F,
     border = T,
     show_rownames=F,
     labels_col = rep('',length(Type[,1])),
     gaps_col = c(322,346,415),
     \## fontsize = 10,
)


```



### Survival analysis

```R
library(survival)
library(survminer)
library(ggplot2)
library(readr)
library(dplyr)
library(extrafont)
library(extrafontdb)
P_val <- read.table("D:/preSolve2/cox/patient_32.txt",header = T)
P_val <- arrange(P_val , SUBTYPE)
patient_information <- P_val
for(i in 1:351) {
 if(patient_information[i,"AGE"] > 64)
   patient_information[i,"AGE"] = ">=65"
 else patient_information[i,"AGE"] = "<65"
}
patient_information <- within(patient_information,{
 SUBTYPE <- factor(SUBTYPE,labels = c("CIN","GS","MSI","EBV"))
 SEX <- factor(SEX,labels = c("female","male"))
 })
patient_information$OS_MONTHS=as.numeric(patient_information$OS_MONTHS)
patient_information$OS_STATUS=as.numeric(patient_information$OS_STATUS)
cox_pict1 <- coxph(Surv(OS_MONTHS , OS_STATUS) ~ SUBTYPE + AGE + SEX , data = patient_information)
ggforest(cox_pict1,
     main = 'CBCS(A)',
     fontsize = 0.8
     )
```



 

### Difference test

The following is the code for selecting genes with significant differences in copy number variation among different subtypes.

```R
library(readr)
F_tot <- read_table("D:/preSolve2/diff_check/New_pat_esca.txt")
F_tot <- as.data.frame(F_tot)
rownames(F_tot) <- F_tot[,1]
F_tot <- F_tot[,-1]
N = 77 ;
T_cna <- F_tot[,c(112:142)]
T_cna <- cbind(F_tot[,1],T_cna)
colnames(T_cna)[1] <- "SUBTYPE"
cna_name <- colnames(T_cna[,c(2:32)])
Tc1 = T_cna
for (i in 1 : N) {
 if(Tc1[i,1] == 1)
  Tc1[i,1] = 1
 else Tc1[i,1] = 2
}
library(dplyr)
p1val = array(0,dim=c(31,2))
p1val = as.data.frame(p1val)
cnt = 0
for(gene in cna_name){
 p = t.test( Tc1[,gene]~Tc1$SUBTYPE )$p.value
 cnt = cnt + 1 ;
 p1val[cnt,1] = gene
 p1val[cnt,2] = p
}
colnames(p1val)[1] <- "Gene"
colnames(p1val)[2] <- "P"
P_cna <- arrange(p1val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_CIN_esca.txt",sep='\t')
\####################################
Tc2 = T_cna
for (i in 1 : N) {
 if(Tc2[i,1] == 2)
  Tc2[i,1] = 1
 else Tc2[i,1] = 2
}
library(dplyr)
p2val = array(0,dim=c(31,2))
p2val = as.data.frame(p2val)
cnt = 0
for(gene in cna_name){
 p = t.test( Tc2[,gene]~Tc2$SUBTYPE )$p.value
 cnt = cnt + 1 ;
 p2val[cnt,1] = gene
 p2val[cnt,2] = p
}
colnames(p2val)[1] <- "Gene"
colnames(p2val)[2] <- "P"
P_cna <- arrange(p2val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_GS_esca.txt",sep='\t')
\######################################
Tc3 = T_cna
for (i in 1 : N) {
 if(Tc3[i,1] == 3)
  Tc3[i,1] = 1
 else Tc3[i,1] = 2
}
library(dplyr)
p3val = array(0,dim=c(31,2))
p3val = as.data.frame(p3val)
cnt = 0
for(gene in cna_name){
 p = t.test( Tc3[,gene]~Tc3$SUBTYPE )$p.value
 cnt = cnt + 1 ;
 p3val[cnt,1] = gene
 p3val[cnt,2] = p
}
colnames(p3val)[1] <- "Gene"
colnames(p3val)[2] <- "P"
P_cna <- arrange(p3val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_MSI_esca.txt",sep='\t')
\################################
Tc4 = T_cna
for (i in 1 : N) {
 if(Tc4[i,1] == 4)
  Tc4[i,1] = 1
 else Tc4[i,1] = 2
}
library(dplyr)
p4val = array(0,dim=c(31,2))
p4val = as.data.frame(p4val)
cnt = 0
for(gene in cna_name){
 p = t.test( Tc4[,gene]~Tc4$SUBTYPE )$p.value
 cnt = cnt + 1 ;
 p4val[cnt,1] = gene
 p4val[cnt,2] = p
}
colnames(p4val)[1] <- "Gene"
colnames(p4val)[2] <- "P"
P_cna <- arrange(p4val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_EBV_esca.txt",sep='\t')
```

Ultimately, the outcomes were subjected to correction via the false discovery rate (FDR), following which statistically significant genes were discerned.

```R
library(readr);
Datar <- read_table("D:/preSolve2/选取差异基因/mely_EBV.txt",col_names = c("id","Gene","value")) ;
Datar <- Datar[,-1];
Datar <- Datar[-1,];
for(i in 1:nrow(Datar[,1])){
 x = Datar[i,1];
 x = as.character(x);
 y = gsub('["]','',x);
 Datar[i,1] = y;
}
tmp <- as.vector(Datar[,2]);
Tp <- as.numeric(unlist(tmp));
v <- p.adjust( Tp , method = "BH" )
sz = 0 ;
for(i in 1:length(v)) {
 if(v[i] < 1e-5)
  sz = i ;
}
v <- v[c(1:sz)]
write.table(cbind(Datar[(1:sz),1],v),file = "D:/preSolve2/选取差异基因/mely_EBV显著基因的p值.txt",     sep='    ',col.names = FALSE,quote=FALSE)
```

