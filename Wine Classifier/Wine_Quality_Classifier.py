



import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


#Function reads dataset from csv file
def readData():

    #Read file
    file = 'winequality-red.csv'
    balance_data = pd.read_csv(file) 

    #Print Dataset
    print("Dataset: \n")
    print(balance_data)
    print("\n")

    return balance_data

#Function splits dataset
def splitSet(balance_data):
    
    #Separating input and output
    X=balance_data.values[:,0:10]
    Y=balance_data.values[:,11]
    
    #Separate training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
    
    return X, Y, X_train, X_test, y_train, y_test
    
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 

    # Performing training 
    clf_gini.fit(X_train, y_train) 
    
    return clf_gini

# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 

    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 3, min_samples_leaf = 5) 

    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    
    return clf_entropy

#Function preforms training with random forest analysis
def train_using_rfc(X_train, X_test, y_train):
    
    #Random Forest Classifier object
    rfc = RandomForestClassifier(n_estimators=200)
    
    #Preform Training
    rfc.fit(X_train, y_train)
    
    return rfc

#Function preforms training with stochastic gradient decent classifier 
def train_using_sgd(X_train, X_test, y_train):
    
    #Stochastic Gradient Decent Classifier
    sgd = SGDClassifier(penalty=None)
    
    #Preform Training
    sgd.fit(X_train, y_train)
    
    return sgd

#Functions preforms training with support vector classifier
def train_using_svc(X_train, X_test, y_train):
    
    #Support Vector Classifier
    svc = SVC()
    
    #Preform Training
    svc.fit(X_train, y_train)
    
    return svc
    
# Function to make predictions 
def prediction(X_test, clf_object): 

    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 

# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 

    print("Confusion Matrix: \n", 
        confusion_matrix(y_test, y_pred)) 

    print ("Accuracy : \n", 
    accuracy_score(y_test,y_pred)*100) 

    print("Report : \n", 
    classification_report(y_test, y_pred)) 
    
#Normalize Data
def normalize(data):
    
    #Normalization with a l1 method
    l1 = preprocessing.normalize(data, norm='l1')
    
    #Normalization with a l2 method
    l2 = preprocessing.normalize(data, norm='l2')
    
    return l1, l2

# Driver code 
def main(): 

    # Building Phase 
    #Obtaining Data
    data = readData()
    
    #Normalized data sets
    #data_l1, data_l2 = normalize(data)
    
    #Split Data
    #Original Data
    X, Y, X_train, X_test, y_train, y_test = splitSet(data)
    #L1 Data
    #X, Y, X_train, X_test, y_train, y_test = splitSet(data_l1)
    #L2 Data
    #X, Y, X_train, X_test, y_train, y_test = splitSet(data_l2)
    
    #Standard Scaler used to pre-process data
    #sc = StandardScaler()
    
    #Scale data
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.fit_transform(X_test)
    
    
    #Training Trees
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 
    rfc = train_using_rfc(X_train, X_test, y_train)
    sgd = train_using_sgd(X_train, X_test, y_train)
    svc = train_using_svc(X_train, X_test, y_train)

    # Operational Phase 
    print("Results Using Gini Index:") 
    
    # Prediction using Gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
    
    # Prediction using Entropy
    print("Results Using Entropy:")  
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    # Prediction using Random Forest Classifier
    print("Results Using Random Forest Classifier:")  
    y_pred_rfc = prediction(X_test, rfc) 
    cal_accuracy(y_test, y_pred_rfc)
    
    # Prediction using Stochastic Gradient Decent Classifier
    print("Results Using Stochastic Gradient Decent Classifier:")  
    y_pred_sgd = prediction(X_test, sgd) 
    cal_accuracy(y_test, y_pred_sgd)
    
    # Prediction using Support Vector Classifier
    print("Results Using Support Vector Classifier:") 
    y_pred_svc = prediction(X_test, svc) 
    cal_accuracy(y_test, y_pred_svc)
     
    
# Calling main function 
if __name__=="__main__": 
    main() 







