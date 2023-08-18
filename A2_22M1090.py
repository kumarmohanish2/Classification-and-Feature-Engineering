# -*- coding: utf-8 -*-
"""Copy_of_Assignment_2_ML_Feature_Engineering.ipynb





#Assignment – 2 : Classification and Feature Engineering

**OBJECTIVE 01:**
"""



"""**1. Let your code read the data directly from https://www.ee.iitb.ac.in/~asethi/Dump/MouseTrain.csv**"""

import numpy as np
import pandas as pnd
import os
# Data is been read from the MouseTrain.csv Given in the above link
Data =pnd.read_csv('https://www.ee.iitb.ac.in/~asethi/Dump/MouseTrain.csv') 
# To print data just write Data
Data

"""Reference for Q:01:- https://www.geeksforgeeks.org/ways-to-import-csv-files-in-google-colab/

**2. Perform exploratory data analysis to find out:**

   a. Which variables are usable, and which are not?

   b. Are there significant correlations among variables?

   c. Are the classes balanced?
"""

# X_data is the feature matrix after dropping the dependent variables
X_data = Data.drop(['Genotype', 'Treatment_Behavior'], axis = 1)
#y_data1 is the First target variable 
y_data1 = Data['Genotype']     
#y_data2 is the First target variable                                   
y_data2 = Data['Treatment_Behavior']   
#The below command print the Remaining data after dropping both dependent variab                         
# X_data

y_data1              # Printing the "Genotype" Target variable                                           # first target variable

y_data2        # Printing the "Treatment_Behavior" Target variable                                                # second target variable

import seaborn as sns
# Finding the correlation matrix for the features of the Data matrix X_Data
corr_Matrix = X_data.corr()
corr_Matrix

def Find_Correlated_Features(data_matrix, threshold): #Function to calculate the Correlated feature                                     # Define correlation function
    corr_feature = set()  #an empty set is created
    corr_matrix = data_matrix.corr()   # Correlation Matrix is formed
    #The below for loop is finding correlation of each feature with another feature and storing the Correlated features ehich have correlation above threshold  
    for i in range(len(corr_matrix)):  
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # Comparing the Correlation of features with the threshold
                feature = corr_matrix.columns[i]  #Retrieving the correlated feauture above threshold From the Data matrix
                corr_feature.add(feature)    #Adding Correlated Feature above threshold
    return corr_feature    # Function returning the Correlated feautures above threshold

# Calling the function to return the feature which are correlated above the threshold value of 0.9
corr_features = Find_Correlated_Features(X_data, 0.9)

print('Number of highly correlated features:', len(corr_features))
print(corr_features)

#Removing the 'pERK_N' feature from the Correlated features
corr_features.remove('pERK_N')
print(corr_features)

# dropping here 6 out of 7 highly correlated features
X_data_a = Data.drop(['Genotype', 'Treatment_Behavior','pS6_N', 'ITSN1_N', 'BRAF_N', 'pNR1_N', 'Bcatenin_N', 'pNR2B_N'], axis = 1)
X_data_a    # Printing the Data matrix

"""Reference:-https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python"""

# Here the function drop_Null is created to drop the features which are having more than 15 null values
def drop_Null(Data1_mat, threshold):   # Function drop_Null is defined
    Null_count = Data1_mat.isna().sum() # Null_Count is the array in which the no. of Null values in each column is stored
    Null_Feature = set()  #Null_Feature is the empty set created
    for i in range(len(Null_count)):   # For loop is run upto the length of the Null_count 
        if Null_count[i] > threshold: # Checking for is column that if no. of null values is greater than some threshold
            Null_Feature.add(Data1_mat.columns[i])  # if condition is satisfied then adding that perticular feature to Null_Feature
        
    print('Features or the columns od the Data matrix which contain more than', threshold, 'null values:', Null_Feature)
    Data1_mat = Data1_mat.drop(Null_Feature, axis = 1)     # dropping the features having more 1 null values
    print('Final Data Matrix after removing the features:')
    return Data1_mat,Null_Feature    # Returning the final data matrix and the Null feature which are dropped
            
X_data = drop_Null(X_data_a, 15)
X_data

"""References:- Took Help from one of my Friend Nayan to remove features with more null values , Roll no.:-22M1200"""

Data['Genotype'].value_counts()

Data['Treatment_Behavior'].value_counts()

"""A) Which variables are usable, and which are not?


> Variables which are highly correlated lets say more than 0.9 and the variables having null values greater than 15 . All these above mentioned variables are not usable.


> Not usable variables are:-'pNR1_N', 'pNR2B_N', 'ITSN1_N', 'BRAF_N', 'Bcatenin_N', 'pS6_N','BAD_N', 'BCL2_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'pCFOS_N'. 


> Other than the variables shown above are usables.



B) Are there significant correlations among variables?

> I have checked on the data for the correlation above 0.9. then it comes out to be that 7 following features are highly correlated.

> {'pNR1_N', 'pNR2B_N', 'ITSN1_N', 'BRAF_N', 'pERK_N', 'Bcatenin_N', 'pS6_N'}


C) Are the classes balanced?

> For both the genotype and the treatment behaviour we can say that classes are balanced. because there is no significant difference between the no. of elements in each classes.

**3. Develop a strategy to deal with missing variables. You can choose to impute the variable. The recommended
way is to use multivariate feature imputation** 


> Comment:- For Imputing using multivariate feature imputation we have used the  inbuilt Iterative imputer.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
 
# Multi variate Imputing using iterated imputer is done 
imputed = IterativeImputer(max_iter=10, random_state=0)     # Iterative Imputer is used 
imputed.fit(X_data_a)                  # Imputing the X_data matrix 
imputed_data = pnd.DataFrame(data=imputed.transform(X_data_a), columns=X_data_a.columns)       # Storing data in the variable named imputed_data 
imputed_data.isnull().sum()    # Checking if there is any null values      
Data_read = imputed_data

"""References:-https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

**4. Select metrics that you will use, such as accuracy, F1 score, balanced accuracy, AUC etc. Remember, you
have two separate classification tasks – one is binary, the other has four classes. You may have to do some
reading about multi-class classification metrics.**

>I have considered the accuracy for binary and F1 score for multi class as the metric

**5. Using five-fold cross-validation (you can use GridSearchCV from scikit-learn) to find the reasonable (I cannot
say “best” because you have two separate classifications to perform) hyper-parameter settings for the
following model types:**

   a. Linear SVM with regularization as hyperparameter 

   b. RBF kernel SVM with kernel width and regularization as hyperparameters

   c. Neural network with single ReLU hidden layer and Softmax output (hyperparameters: number of
neurons, weight decay)

   d. Random forest (max tree depth, max number of variables per node)**

**Binary Classification** for Genotype,
Accuracy is metric here
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X= Data_read  #Data_read is the Data after feature selection and imputation
model = LinearSVC()    # Here Model is Defined and initialized as the LinearSVC() 
y1 = Data['Genotype']   # One predicted variable is Extracted from the Data

X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.4)  # We spitting the Data matrix into the training and Testing Data
model.fit(X_train, Y_train)  #Using Training Data we are training the model

Y_pred = model.predict(X_test)  # Here using test data precticting the model output 

accurcy = accuracy_score(Y_test, Y_pred)    # Calculating Model's Accuracy score Using In built Library of the model on test data

# Printing the accuracy score
# print("Accuracy:", accurcy)

# Here In the below four line defining hyperparameter grids for all four modes respectively
param_grid_linear_svm = {'C': [0.01, 0.1, 1, 10]}    # Hyperparameter grids for Linear SVM
param_grid_rbf_svm = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}     # Hyperparameter grids for RBF Kernel SVM
param_grid_neural_network = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.001, 0.01]}       # Hyperparameter grids for Neural Network
param_grid_random_forest = {'max_depth': [5, 10, 15], 'max_features': ['sqrt', 'log2', None]}      # Hyperparameter grids for Random Forest

# Here In below Four lines the four models are defiened
linear_svm = LinearSVC(max_iter=1000)       
rbf_svm = SVC(kernel='rbf')
neural_network = MLPClassifier(activation='relu', solver='adam', random_state=10,max_iter=100)
random_forest = RandomForestClassifier(random_state=10)

# Here Performing the Cross-Validation with grid search
grid_search_linear_svm = GridSearchCV(linear_svm, param_grid_linear_svm, cv=5, scoring='accuracy')
grid_search_rbf_svm = GridSearchCV(rbf_svm, param_grid_rbf_svm, cv=5, scoring='accuracy')
grid_search_neural_network = GridSearchCV(neural_network, param_grid_neural_network, cv=5, scoring='accuracy')
grid_search_random_forest = GridSearchCV(random_forest, param_grid_random_forest, cv=5, scoring='accuracy')

# Here Fitting models and printing best hyperparameters
grid_search_linear_svm.fit(X_train, Y_train)
print("Linear SVM: Best hyperparameters:", grid_search_linear_svm.best_params_)
grid_search_rbf_svm.fit(X_train, Y_train)
print("RBF SVM: Best hyperparameters:", grid_search_rbf_svm.best_params_)
grid_search_neural_network.fit(X_train, Y_train)
print("Neural Network: Best hyperparameters:", grid_search_neural_network.best_params_)
grid_search_random_forest.fit(X_train, Y_train)
print("Random Forest: Best hyperparameters:", grid_search_random_forest.best_params_)

"""References:- https://www.youtube.com/watch?v=HdlDYng8g9s&ab_channel=codebasics



**Multiclass Classification** for Treatment_Behavior,
 **F1 score** is metric here
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

X= Data_read  #Data_read is the Data after feature selection and imputation
model = LinearSVC()    # Here Model is Defined and initialized as the LinearSVC() 
y2 = Data['Treatment_Behavior']   # One predicted variable is Extracted from the Data

X_train, X_test, Y_train, Y_test = train_test_split(X, y2, test_size=0.2)  # We spitting the Data matrix into the training and Testing Data
model.fit(X_train, Y_train)  #Using Training Data we are training the model

Y_pred = model.predict(X_test)  # Here using test data precticting the model output 

# accurcy = accuracy_score(Y_test, Y_pred)    # Calculating Model's Accuracy score Using In built Library of the model on test data

# Printing the accuracy score
# print("Accuracy:", accurcy)

# Here In the below four line defining hyperparameter grids for all four modes respectively
param_grid_linear_svm = {'C': [0.01, 0.1, 1, 10]}    # Hyperparameter grids for Linear SVM
param_grid_rbf_svm = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}     # Hyperparameter grids for RBF Kernel SVM
param_grid_neural_network = {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.001, 0.01]}       # Hyperparameter grids for Neural Network
param_grid_random_forest = {'max_depth': [5, 10, 15], 'max_features': ['sqrt', 'log2', None]}      # Hyperparameter grids for Random Forest

# Here In below Four lines the four models are defiened
linear_svm = LinearSVC(max_iter=1000)       
rbf_svm = SVC(kernel='rbf')
neural_network = MLPClassifier(activation='relu', solver='adam', random_state=10,max_iter=100)
random_forest = RandomForestClassifier(random_state=10)

# Here Performing the Cross-Validation with grid search
grid_search_linear_svm = GridSearchCV(linear_svm, param_grid_linear_svm, cv=5, scoring='f1_weighted')
grid_search_rbf_svm = GridSearchCV(rbf_svm, param_grid_rbf_svm, cv=5, scoring='f1_weighted')
grid_search_neural_network = GridSearchCV(neural_network, param_grid_neural_network, cv=5, scoring='f1_weighted')
grid_search_random_forest = GridSearchCV(random_forest, param_grid_random_forest, cv=5, scoring='f1_weighted')

# Here Fitting models and printing best hyperparameters
grid_search_linear_svm.fit(X_train, Y_train)
print("Linear SVM: Best hyperparameters:", grid_search_linear_svm.best_params_)
grid_search_rbf_svm.fit(X_train, Y_train)
print("RBF SVM: Best hyperparameters:", grid_search_rbf_svm.best_params_)
grid_search_neural_network.fit(X_train, Y_train)
print("Neural Network: Best hyperparameters:", grid_search_neural_network.best_params_)
grid_search_random_forest.fit(X_train, Y_train)
print("Random Forest: Best hyperparameters:", grid_search_random_forest.best_params_)

"""**6. Check feature importance for each model to see if the same proteins are important for each model. Read upon how to find feature importance**

**Linear SVM**

> Binary Classification
"""

# permutation feature importance with Linear SVM for Binary classification
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

# define the model
model = LinearSVC()
# fit the model
model.fit(X, y1)
# perform permutation importance
results = permutation_importance(model, X, y1, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""

> Multi Classification

"""

# permutation feature importance with Linear SVM for Multi classification
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

# define the model
model = LinearSVC()
# fit the model
model.fit(X, y2)
# perform permutation importance
results = permutation_importance(model, X, y2, scoring='f1_weighted')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""**RBF kernel SVM**

Binary Classification
"""

# permutation feature importance with RBF kernel SVM for Binary classification
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

# Here in below two line fitting RBF_SVM model on data
rbf_SVM = SVC(kernel='rbf')
rbf_SVM.fit(X,y1) # using SVM model on data matrix X and the for binary classification


# Here in below two lines also performing the permutation importance
results = permutation_importance(rbf_SVM, X, y1)    
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plotting the feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""Multi Classification"""

# permutation feature importance with RBF kernel SVM for Multi classification
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot


# Here in below two line fitting RBF_SVM model on data
rbf_SVM = SVC(kernel='rbf')
rbf_SVM.fit(X,y2) # using SVM model on data matrix X and the for binary classificatio


# Here in below two lines also performing the permutation importance
results = permutation_importance(rbf_SVM, X, y2)
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# ploting feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""**Neural Network**

> Binary Classification
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

# Creating Neural network object
neural_network = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=10, max_iter=100)

# Fitting The neural network on the data
neural_network.fit(X,y1)
#Performing the permutation importance on the data
results = permutation_importance(neural_network, X, y1)
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""

> Multi Classification

"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

# Creating Neural network object
neural_network = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=10, max_iter=100)

# Fitting The neural network on the data
neural_network.fit(X,y2)
#Performing the permutation importance on the data
results = permutation_importance(neural_network, X, y2)
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""**Random_Forest**

Binary Classification
"""

# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y1)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""Multi Classification"""

# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y2)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

"""References For Q:06:-

> https://machinelearningmastery.com/calculate-feature-importance-with-python/

> https://www.youtube.com/watch?v=R47JAob1xBY&ab_channel=CampusX

**Q:07 See if removing some features systematically will improve your models (e.g. using recursive feature elimination https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html). [3]**

**Linear_SVM**

For Binary Class
"""

from sklearn.feature_selection import RFE

# Defining the Model
linear_svm = LinearSVC(max_iter=1000,C=10)
estimator=linear_svm
rfe1 = RFE(estimator, step=1)               # recursive feature elimination

X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.2)     # Splittiong the data into Testing and Traing data where training data is 80 percent while testing data is 20 percent

# fitting the RFE Object into the Splitted Training Data 
rfe1.fit(X_train, Y_train)    

print(rfe1.ranking_)    # Print the ranking of each feature
print(rfe1.support_)  # Tells whether the feature is eliminated or not


X_train_new1 = rfe1.transform(X_train)   # Reduce the unwanted feature from the training data
X_test_new = rfe1.transform(X_test)     # Reduce the unwanted feature from the testing data
model.fit(X_train_new1, Y_train)       # fitting the model for the reduced training data with Y_train
Y_pred = model.predict(X_test_new)     # Predicting the data for reduced testing data
Y_train1 = Y_train                 
accuracy = accuracy_score(Y_test, Y_pred)   # Evaluating the Accuracy

# Print the accuracy score
print("Accuracy:", accuracy)

"""

> For Multiclass

"""

from sklearn.feature_selection import RFE

# Defining the Model
linear_svm = LinearSVC(max_iter=1000,C=10)
estimator=linear_svm
rfe2 = RFE(estimator, step=1)               # recursive feature elimination

X_train, X_test, Y_train, Y_test = train_test_split(X, y2, test_size=0.2)     # Splittiong the data into Testing and Traing data where training data is 80 percent while testing data is 20 percent

# fitting the RFE Object into the Splitted Training Data 
rfe2.fit(X_train, Y_train)    

print(rfe2.ranking_)    # Print the ranking of each feature
print(rfe2.support_)  # Tells whether the feature is eliminated or not


X_train_new2 = rfe2.transform(X_train)   # Reduce the unwanted feature from the training data
X_test_new = rfe2.transform(X_test)     # Reduce the unwanted feature from the testing data
model.fit(X_train_new2, Y_train)       # fitting the model for the reduced training data with Y_train
Y_pred = model.predict(X_test_new)     # Predicting the data for reduced testing data
Y_train2 = Y_train
accuracy = accuracy_score(Y_test, Y_pred)   # Evaluating the Accuracy

# Print the accuracy score
print("Accuracy:", accuracy)

"""REfERENCES:-

> https://www.blog.trainindata.com/recursive-feature-elimination-with-python/


> https://www.geeksforgeeks.org/recursive-feature-elimination-with-cross-validation-in-scikit-learn/

**Random_Forest**

> For Binary CLass
"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X,y1, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn. feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn. feature_selection import RFE


# Performing RFE using for Random FOrest Classication 
sel = RFE (RandomForestClassifier (n_estimators=100, random_state=0, n_jobs=-1), n_features_to_select = 15)

# Fitting the Model
sel.fit(X_train, Y_train)

sel.fit(X_train, Y_train)
sel.get_support()

X_train.columns

features = X_train.columns[sel.get_support()]
features

#Lenth of the features which are remained
len(features)

np.mean(sel. estimator_.feature_importances_)

sel.estimator_.feature_importances_

# Transforming X_train and X_test with recursive feature elimination
x_train_rfe = sel.transform(X_train)
x_test_rfe = sel.transform(X_test)

# Defining a function to find the accuracy after the Random forst classification in the remained data
def run_randomForest(X_train, X_test, Y_train, Y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(Y_test, y_pred))

run_randomForest(x_train_rfe, x_test_rfe, Y_train, Y_test)

"""For MultiClass"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X,y2, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn. feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn. feature_selection import RFE


# Performing RFE using for Random FOrest Classication 
sel = RFE (RandomForestClassifier (n_estimators=100, random_state=0, n_jobs=-1), n_features_to_select = 15)

# Fitting the Model
sel.fit(X_train, Y_train)

sel.fit(X_train, Y_train)
sel.get_support()

X_train.columns

features = X_train.columns[sel.get_support()]
features

#Lenth of the features which are remained
len(features)

np.mean(sel. estimator_.feature_importances_)

sel.estimator_.feature_importances_

# Transforming X_train and X_test with recursive feature elimination
x_train_rfe = sel.transform(X_train)
x_test_rfe = sel.transform(X_test)

# Defining a function to find the accuracy after the Random forst classification in the remained data
def run_randomForest(X_train, X_test, Y_train, Y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(Y_test, y_pred))

run_randomForest(x_train_rfe, x_test_rfe, Y_train, Y_test)

"""REFERENCES:-

>https://www.youtube.com/watch?v=pcZ4YlvhSKU&ab_channel=KGPTalkie

**8. Finally, test a few promising models on the test data:
https://www.ee.iitb.ac.in/~asethi/Dump/MouseTest.csv**

**Linear SVM** , For Multi Class
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import LinearSVC, SVC

url = 'https://www.ee.iitb.ac.in/~asethi/Dump/MouseTest.csv' # URL of the CSV file to read
Data = pd.read_csv(url) # Read the CSV file into a DataFrame
Test_Data = Data.drop(columns=['Genotype' , 'Treatment_Behavior'])
Y1_test = Data['Treatment_Behavior']
Y2_test = Data['Genotype']

X_data_test = Test_Data.drop(['pS6_N', 'ITSN1_N', 'BRAF_N', 'pNR1_N', 'Bcatenin_N', 'pNR2B_N'], axis = 1)
X_data_final = X_data_test.drop(['H3MeK4_N', 'BCL2_N', 'BAD_N', 'H3AcK18_N', 'EGR1_N', 'pCFOS_N'], axis=1)
 

imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(X_data_final)
data_imputed_test = pd.DataFrame(data=imp.transform(X_data_final), columns=X_data_final.columns)

estimator = LinearSVC(max_iter=1000)
x_test = rfe2.transform(data_imputed_test)
estimator.fit(X_train_new2, y_train2)   # model fit with reduced features
y_pred_test = estimator.predict(x_test) # predicting the y vector from data and validation data

# Evaluate the model's accuracy on the test data
accuracy = accuracy_score(Y1_test, y_pred_test)

# Print the accuracy score
print("Accuracy:", accuracy)

"""# Objective 2

**9. Read the pytorch tutorial to use a pre-trained “ConvNet as fixed feature extractor” fromhttps://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html and you can ignore “finetuning theConvNet”. Test this code out to see if it runs properly in your environment after eliminating code blocks thatyou do not need. [2]**
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import zipfile
zip_ref = zipfile.ZipFile('/content/hymenoptera_data.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

"""Training the model

"""

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

"""Visualizing the model predictions"""

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

"""ConvNet as fixed feature extractor

"""

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

"""Train and evaluate"""

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=2)

visualize_model(model_conv)

plt.ioff()
plt.show()

"""Reference:-https://www.youtube.com/watch?v=pcZ4YlvhSKU&ab_channel=KGPTalkie

**10. Write a function that outputs ResNet18 features for a given input image. Extract features for training images(in image_datasets['train']). You should get an Nx512 dimensional array.**
"""

from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def extract_resnet_features(image_path):
    # Load the pre-trained ResNet18 model
    resnet = models.resnet18(pretrained=True)

    # Remove the last fully connected layer
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)

    # Set the model to evaluation mode
    resnet.eval()

    # Load the image and apply the appropriate transformations
    image = Image.open('/content/hymenoptera_data/train/ants/0013035.jpg')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    # Extract the features for the image
    with torch.no_grad():
        features = resnet(image)

    # Return the features as a numpy array
    return features.squeeze().numpy()

features = []
for inputs, labels in dataloaders['train']:
    for img_tensor in inputs:
        img_features = extract_resnet_features(img_tensor)
        features.append(img_features)

features = np.array(features)
print(features.shape)  # should output (N, 512)




