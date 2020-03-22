#Support Vector Machines
import numpy as np
import matplotlib.pyplot as mp
from pylab import show
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import statistics
from sklearn.ensemble import AdaBoostClassifier
import time

data = pd.read_csv("BCV.csv")

np.random.seed(100)

X, y = data.iloc[:,1:10],data.iloc[:,10]
trainFeatures, testFeatures, trainDigits, testDigits = train_test_split(X, y, test_size=0.2, random_state=123)

#select the proportion of data to use for training
CVE = []
C = []
for xp in range (0,10):
    Cp = 0.01* (1.45**xp)
    model = SVC(C=Cp, kernel ='linear', gamma='auto')
    modelL = model.fit(trainFeatures, trainDigits)
    CVE.append(1 - statistics.mean(cross_val_score(modelL,trainFeatures,trainDigits, cv=10,scoring='accuracy')))
    C.append(Cp)
    
mp.xscale('log')
mp.scatter(C,CVE)
mp.xlabel('log(C)')
mp.ylabel('CVE')
mp.title(' SVM Linear , CVE vs Log(C)')
Cval = C[np.argmin(np.asarray(CVE))]
print("Value of C:",Cval)
show()
#### neural network
CVE = []
hidlay = [1,2,5]
numnodes = [[2], [5], [10],[50]]

for i in hidlay:
    for j in numnodes:
        size =i*j
        print(tuple(size))
        mlp = MLPClassifier(hidden_layer_sizes=(tuple(size)), activation='relu' , max_iter=10000, alpha=0, solver='adam', epsilon=0.001)
        CVE.append(1 - np.mean(cross_val_score(mlp,trainFeatures,trainDigits, cv=10,scoring='accuracy')))
print("10-fold CVE =",CVE)
###random forest

n_est = [5,10, 15, 20,30,40,50,75,100,200,500]
CFVE=[]
var = []
for i in range(len(n_est)):
    model = RandomForestClassifier(criterion='entropy',n_estimators=n_est[i], max_leaf_nodes=100)
    CV =cross_val_score(model,trainFeatures,trainDigits, cv=10,scoring='accuracy')
    CFVE.append(1 - statistics.mean(CV))

n_estval = n_est[np.argmin(np.asarray(CFVE))]    
mp.figure(1)
mp.xscale('log')
mp.xlabel('Log(n_estimator)')
mp.ylabel('Cross Validation Error')
mp.title('Fig 4.5, CVE vs n_estimator')
mp.scatter(n_est,CFVE,alpha=.8)
show()
print("Value of n_estimator:",n_estval)

start_time = time.time()

svm = SVC(C=Cval, kernel ='linear', gamma = 'auto')
svm_one = svm.fit(trainFeatures, trainDigits)

#for a in range(len(testDigits)):
tpred_svm=svm_one.predict(testFeatures)

error_svm = 1 - accuracy_score(testDigits,tpred_svm)
print("Error for SVM = ",error_svm)
runtime = time.time() - start_time
print("runtime", "--- %s seconds ---" %runtime)

start_time = time.time()
model_rf = RandomForestClassifier(criterion = "entropy", n_estimators = n_estval, max_leaf_nodes = 1000)
model_rf.fit(trainFeatures, trainDigits)
tpred_rf=model_rf.predict(testFeatures)

error_rf = 1 - accuracy_score(testDigits,tpred_rf)
print("Error for RF = ",error_rf)
runtime = time.time() - start_time
print("runtime", "--- %s seconds ---" %runtime)

start_time = time.time()
mlp_model = MLPClassifier(hidden_layer_sizes=(5), activation='relu' , max_iter=10000, alpha=0, solver='adam', epsilon=0.001, learning_rate_init = 0.001)
mlp_model.fit(trainFeatures,trainDigits)
tpred_mlp=mlp_model.predict(testFeatures)

error_mlp = 1 - accuracy_score(testDigits,tpred_mlp)
print("Error for mlp = ",error_mlp)
runtime = time.time() - start_time
print("runtime", "--- %s seconds ---" %runtime)


