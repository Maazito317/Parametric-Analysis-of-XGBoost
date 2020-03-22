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
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head()
data['PRICE'] = boston.target
data.describe()

X, y = data.iloc[:,:-1],data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

train_matrix = xgb.DMatrix(data=X_train,label=y_train)
#Finding optimal value for max depth
n_estimators=[1,10,20,40,60,80,100,200,300,400,500,1000,1500]
CFVE=[]

for m in range (len(n_estimators)):
    params = {"objective":"reg:linear",'colsample_bytree':1,'learning_rate': 0.3,'gamma':0,
                'max_depth': 8, 'min_child_weight': 1,'subsample':1}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round = n_estimators[m],
                    metrics="rmse", as_pandas=True, seed=27)

    CFVE.append(Cv["test-rmse-mean"].min())
mp.figure(8)
mp.xlabel('n_estimators')
mp.ylabel('RMSE')
mp.title('optimal value of n_estimators')
mp.plot(n_estimators,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_n_estimators = n_estimators[np.argmin(CFVE)]
print("optimal n_estimators", Opt_n_estimators)

max_depth=[1,2,3,4,5,6,7,8,9]
min_child_weight=[1,2,3,4,5,6,7,8,9]
CFVE=[]
for md in range (len(max_depth)):
    for cw in range (len(min_child_weight)):
        params = {"objective":"reg:linear",'colsample_bytree': 1,'learning_rate': 0.3,'gamma':0,
                        'max_depth': max_depth[md],'min_child_weight': min_child_weight[cw],'subsample':1}
        Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round =  Opt_n_estimators,
                             metrics="rmse", as_pandas=True, seed=27)
        CFVE.append(Cv["test-rmse-mean"].min())
cc = np.argmin(np.asarray(CFVE))
print("max depth and min child weight", CFVE)
Opt_max_depth =max_depth[int(cc/len(max_depth))]
Opt_min_child_weight=min_child_weight[int(cc%len(max_depth))]
print(cc,Opt_max_depth,Opt_min_child_weight)

#
gamma=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
CFVE=[]
for m in range (len(gamma)):
    params = {"objective":"reg:linear",'colsample_bytree':1,'learning_rate': 0.3,'gamma':gamma[m],
                'max_depth': Opt_max_depth, 'min_child_weight': Opt_min_child_weight,'subsample':1}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round =  Opt_n_estimators, metrics="rmse", as_pandas=True, seed=27)
    CFVE.append(Cv["test-rmse-mean"].min())


mp.figure(8)
mp.xlabel('gamma')
mp.ylabel('RMSE')
mp.title('optimal value of gamma')
mp.plot(gamma,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_gamma = gamma[np.argmin(CFVE)]
print("optimal gamma", Opt_gamma)

n_estimators=[1,10,20,40,60,80,100,200,300,400,500,1000,1500]
CFVE=[]

for m in range (len(n_estimators)):
    params = {"objective":"reg:linear",'colsample_bytree':1,'learning_rate': 0.3,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'min_child_weight': Opt_min_child_weight,'subsample':1}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,num_boost_round = n_estimators[m],
                    metrics="rmse", as_pandas=True, seed=27)

    CFVE.append(Cv["test-rmse-mean"].min())

mp.figure(8)
mp.xlabel('n_estimators')
mp.ylabel('RMSE')
mp.title('optimal value of n_estimators')
mp.plot(n_estimators,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_n_estimators = n_estimators[np.argmin(CFVE)]
print("optimal n_estimators", Opt_n_estimators)


subsample=[0.1,0.2,0.3,0.5,0.7,1.0]
colsample=[0.1,0.2,0.3,0.5,0.7,1.0]
CFVE=[]
for md in range (len(subsample)):
    for cw in range (len(colsample)):
        params = {"objective":"reg:linear",'colsample_bytree': colsample[cw],'learning_rate': 0.3,'gamma':Opt_gamma,
                        'max_depth': Opt_max_depth, 'min_child_weight': Opt_min_child_weight,'subsample':subsample[md]}
        Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                        num_boost_round=Opt_n_estimators,metrics="rmse", as_pandas=True, seed=27)
        CFVE.append(Cv["test-rmse-mean"].min())
cc = np.argmin(np.asarray(CFVE))
print("sub sample and col sample",CFVE)
Opt_subsample =subsample[int(cc/len(subsample))]
Opt_colsample=colsample[int(cc%len(colsample))]
print(Opt_subsample,Opt_colsample)

###  start of the regularization
alpha=[0,1e-3, 0.005,0.01,0.02,0.004,0.1,0.2,0.3,0.4,1]
CFVE=[]

for m in range (len(alpha)):
    params = {"objective":"reg:linear",'colsample_bytree': Opt_colsample,'learning_rate': 0.3,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':alpha[m],'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=Opt_n_estimators,metrics="rmse", as_pandas=True, seed=27)

    CFVE.append(Cv["test-rmse-mean"].min())
print(CFVE)

mp.figure(2)
mp.xlabel('alpha')
mp.ylabel('RMSE')
mp.title('optimal value of alpha')
mp.plot(alpha,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_alpha = alpha[np.argmin(CFVE)]
print(Opt_alpha)

lambda_reg=[0,1e-3, 0.005,0.01,0.02,0.004,0.1,0.2,0.3,0.4,1]
CFVE=[]

for m in range (len(lambda_reg)):
    params = {"objective":"reg:linear",'colsample_bytree': Opt_colsample,'learning_rate': 0.3,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':Opt_alpha, 'reg_lambda' : lambda_reg[m], 'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=Opt_n_estimators,metrics="rmse", as_pandas=True, seed=27)

    CFVE.append(Cv["test-rmse-mean"].min())
print(CFVE)

mp.figure(2)
mp.xlabel('lambda')
mp.ylabel('RMSE')
mp.title('optimal value of lambda')
mp.plot(lambda_reg,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_lambda = lambda_reg[np.argmin(CFVE)]
print(Opt_lambda)

learning_rate=[0.01,0.03,0.1,0.15,0.2,0.25,0.3,0.4,0.5]
CFVE=[]

for m in range (len(learning_rate)):
    params = {"objective":"reg:linear",'colsample_bytree': Opt_colsample,'learning_rate': learning_rate[m],'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':Opt_alpha,'reg_lambda' : Opt_lambda, 'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=Opt_n_estimators,metrics="rmse", as_pandas=True, seed=27)

    CFVE.append(Cv["test-rmse-mean"].min())

mp.figure(2)
mp.xlabel('learning rate')
mp.ylabel('')
mp.title('optimal value of learning rate')
mp.plot(learning_rate,CFVE,'bo')
mp.show()
CFVE = np.asarray(CFVE)
Opt_learning_rate = learning_rate[np.argmin(CFVE)]
print(Opt_learning_rate)

n_estimators=[1,10,20,40,60,80,100,200,300,400,500,1000]
CFVE=[]

for m in range (len(n_estimators)):
    params = {"objective":"reg:linear",'colsample_bytree': Opt_colsample,'learning_rate': Opt_learning_rate,'gamma':Opt_gamma,
                'max_depth': Opt_max_depth, 'reg_alpha':Opt_alpha,'reg_lambda' : Opt_lambda,'min_child_weight': Opt_min_child_weight,'subsample':Opt_subsample}
    Cv = xgb.cv(dtrain=train_matrix, params=params, nfold=10,
                    num_boost_round=n_estimators[m],metrics="rmse", as_pandas=True, seed=27)

    CFVE.append(Cv["test-rmse-mean"].min())
print(CFVE)
mp.figure(8)
mp.xlabel('n_estimators')
mp.ylabel('RMSE')
mp.title('optimal value of n_estimators')
mp.plot(n_estimators,CFVE,'bo')
mp.show()
print("optimal n_estimators", Opt_n_estimators)


xg_reg = xgb.XGBRegressor(objective ='reg:linear', n_estimators = Opt_n_estimators, colsample_bytree = Opt_colsample, learning_rate = Opt_learning_rate, gamma = Opt_gamma,
            max_depth = Opt_max_depth, reg_alpha = Opt_alpha, reg_lambda = Opt_lambda, min_child_weight = Opt_min_child_weight, subsample = Opt_subsample)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

#linear regression
reg = LinearRegression().fit(X_train, y_train)
preds1 = reg.predict(X_test)
rmse1 = np.sqrt(mean_squared_error(y_test, preds1))
print("RMSE: %f" % (rmse1))
