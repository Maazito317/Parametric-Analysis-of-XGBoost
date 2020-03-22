import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as mp
import statistics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
data = pd.read_csv("BCV.csv")

#Data processing
X, y = data.iloc[:,1:10],data.iloc[:,10]
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
train_matrix = xgb.DMatrix(data=X_train,label=y_train)

#Finding optimal value for n_estimators
n_estimators=[1,10,20,40,60,80,100,200,300,400,500,1000]
CFVE=[]

for m in range (len(n_estimators)):
    params = {"objective":"binary:logistic",'colsample_bytree':0.8,'learning_rate': 0.3,'gamma':0,
                'max_depth': 5, 'min_child_weight': 1,'subsample':0.8}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round = n_estimators[m],
                    metrics="error", as_pandas=True, seed=27)
    CFVE.append(Cv["test-error-mean"].min()

mp.figure(8)
mp.xlabel('n_estimators')
mp.ylabel('ECV')
mp.title('CVE vs n_estimator')
mp.plot(n_estimators,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_n_estimators = n_estimators[np.argmin(CFVE)]
print("Optimal n_estimators", Opt_n_estimators)

#Finding optimal value for max_depth and min_child_weight
max_depth=[1,2,3,4,5,6,7,8,9]
min_child_weight=[1,2,3,4,5,6,7,8,9]
CFVE=[]
for md in range (len(max_depth)):
    for cw in range (len(min_child_weight)):
        params = {"objective":"binary:logistic",'colsample_bytree': 1,'learning_rate': 0.3,'gamma':0.0,
                        'max_depth': max_depth[md],'min_child_weight': min_child_weight[cw],'subsample':1}
        Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round =  Opt_n_estimators,
                             metrics="error", as_pandas=True, seed=27)
        CFVE.append(Cv["test-error-mean"].min())
cc = np.argmin(np.asarray(CFVE))

Opt_max_depth =max_depth[int(cc/len(max_depth))]
Opt_min_child_weight=min_child_weight[int(cc%len(max_depth))]
print("ECV = ",CFVE)
print("Optimal max depth = ",Opt_max_depth,"Optimal minimum child weight =",Opt_min_child_weight)

#Finding optimal value of gamma
gamma=[0,0.1,0.2,0.3,0.5,0.7,1.0,2,5]
CFVE=[]
for m in range (len(gamma)):
    params = {"objective":"binary:logistic",'colsample_bytree':1,'learning_rate': 0.3,'gamma':gamma[m],
                'max_depth': Opt_max_depth, 'min_child_weight': Opt_min_child_weight,'subsample':1}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round =  Opt_n_estimators, metrics="error", as_pandas=True, seed=27)
    CFVE.append(Cv["test-error-mean"].min())

mp.figure(8)
mp.xlabel('Gamma')
mp.ylabel('ECV')
mp.title('CVE vs Gamma')
mp.plot(gamma,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_gamma = gamma[np.argmin(CFVE)]
print("Optimal gamma", Opt_gamma)

#Reevaluating n_estimators
n_estimators=[1,10,20,40,60,80,100,200,300,400,500,1000]
CFVE=[]

for m in range (len(n_estimators)):
    params = {"objective":"binary:logistic",'colsample_bytree':0.8,'learning_rate': 0.3,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'min_child_weight': Opt_min_child_weight,'subsample':0.8}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round = n_estimators[m],
                    metrics="error", as_pandas=True, seed=27)
    CFVE.append(Cv["test-error-mean"].min())

mp.figure(8)
mp.xlabel('n_estimators')
mp.ylabel('ECV')
mp.title('CVE vs n_estimator')
mp.plot(n_estimators,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_n_estimators = n_estimators[np.argmin(CFVE)]
print("Optimal n_estimators", Opt_n_estimators)

#Optimal value of subsample and colsample_bytree
subsample=[0.1,0.2,0.3,0.5,0.7,1.0]
colsample=[0.1,0.2,0.3,0.5,0.7,1.0]
CFVE=[]
for md in range (len(subsample)):
    for cw in range (len(colsample)):
        params = {"objective":"binary:logistic",'colsample_bytree': colsample[cw],'learning_rate': 0.3,'gamma':Opt_gamma,
                        'max_depth': Opt_max_depth, 'min_child_weight': Opt_min_child_weight,'subsample':subsample[md]}
        Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                        num_boost_round=Opt_n_estimators,metrics="error", as_pandas=True, seed=27)
        CFVE.append(Cv["test-error-mean"].min())
cc = np.argmin(np.asarray(CFVE))
print("ECV = ",CFVE)
Opt_subsample =subsample[int(cc/len(subsample))]
Opt_colsample=colsample[int(cc%len(colsample))]
print("Optimal sub sample = ",Opt_subsample,"optimal col sample by tree",Opt_colsample)

###  start of the regularization L1 and L2
alpha=[0,1e-3, 0.01,0.1,0.2,0.3,0.4,1]
CFVE=[]

for m in range (len(alpha)):
    params = {"objective":"binary:logistic",'colsample_bytree': Opt_colsample,'learning_rate': 0.3,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':alpha[m],'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=Opt_n_estimators,metrics="error", as_pandas=True, seed=27)

    CFVE.append(Cv["test-error-mean"].min())

mp.figure(2)
mp.xlabel('alpha')
mp.ylabel('ECV')
mp.title('CVE vs alpha')
mp.plot(alpha,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_alpha = alpha[np.argmin(CFVE)]
print(Opt_alpha)

lamb=[0,1e-3, 0.01,0.1,0.2,0.3,0.4,1]
CFVE=[]

for m in range (len(lamb)):
    params = {"objective":"binary:logistic",'colsample_bytree': Opt_colsample,'learning_rate': 0.3,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':Opt_alpha, 'reg_lambda': lamb[m], 'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=Opt_n_estimators,metrics="error", as_pandas=True, seed=27)
    CFVE.append(Cv["test-error-mean"].min())

mp.figure(2)
mp.xlabel('lambda')
mp.ylabel('ECV')
mp.title('CVE vs lambda')
mp.plot(alpha,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_lambda = lamb[np.argmin(CFVE)]
print(Opt_lambda)

#Tweaking learning rate
learning_rate=[0.01,0.03,0.1,0.2,0.5,0.8,1.1,1.5,2]
CFVE=[]

for m in range (len(learning_rate)):
    params = {"objective":"binary:logistic",'colsample_bytree': Opt_colsample,'learning_rate': learning_rate[m],'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':Opt_alpha, 'reg_lambda':Opt_lambda, 'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=Opt_n_estimators,metrics="error", as_pandas=True, seed=27)

    CFVE.append(Cv["test-error-mean"].min())

mp.figure(2)
mp.xlabel('learning rate')
mp.ylabel('ECV')
mp.title('Optimal value of learning rate')
mp.plot(learning_rate,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_learning_rate = learning_rate[np.argmin(CFVE)]
print(Opt_learning_rate)

#Final evaluation of n_estimators
n_estimators=[1,10,20,40,60,80,100,200,300,400,500,1000]
CFVE=[]

for m in range (len(n_estimators)):
    params = {"objective":"binary:logistic",'colsample_bytree': Opt_colsample,'learning_rate': Opt_learning_rate,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':Opt_alpha,'reg_lambda':Opt_lambda,'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=n_estimators[m],metrics="error", as_pandas=True, seed=27)

    CFVE.append(Cv["test-error-mean"].min())
print(CFVE)
mp.figure(8)
mp.xlabel('n_estimators')
mp.ylabel('ECV')
mp.title('Optimal value of n_estimators')
mp.plot(n_estimators,CFVE,'bo')
mp.show()
print("Optimal n_estimators", Opt_n_estimators)

###   test

xg_reg = xgb.XGBClassifier(objective ='binary:logistic',n_estimators = Opt_n_estimators,colsample_bytree= Opt_colsample,learning_rate= Opt_learning_rate,gamma=Opt_gamma,
            max_depth= Opt_max_depth, reg_alpha=Opt_alpha,reg_lambda=Opt_lambda, min_child_weight= Opt_min_child_weight,subsample=Opt_subsample)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
print(accuracy_score(y_test,preds))
