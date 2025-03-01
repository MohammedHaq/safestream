# generally used
import pandas as pd
import numpy as np

# used for file size and model saving
import os
import pickle

# used for displaying features
import seaborn as sns
import matplotlib.pyplot as plt

# used for training model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle

# used for data augmentation
from imblearn.over_sampling import SMOTE

# constants for where we save our initial model and our optimized model
initial_model_filename = "models/model.sav"
optimized_model_filename = "models/optimized_model.sav"

# constants for our saved test data files
X_initial = "data/X_initial.pkl"
Y_initial = "data/Y_initial.pkl"
X_optimized = "data/X_optimized.pkl"
Y_optimized = "data/Y_optimized.pkl"


# Method for regular preprocessing of our model before any size optimizations
def preprocess_initial_data():

    # Load data
    data = pd.read_csv('./data/water_potability.csv')

    # Drop rows with missing values
    data = data.dropna()

    # Balance the dataset using equality resampling. This is especially importnat because our model 
    # was heavily skewed towards one class
    notpotable = data[data['Potability'] == 0]
    potable = data[data['Potability'] == 1]
    df_minority_upsampled = resample(potable, replace=True, n_samples=3000)
    data = pd.concat([notpotable, df_minority_upsampled])
    data = shuffle(data, random_state=42)

    # create feature matrix and target vector
    x = data.drop(['Potability'], axis=1)
    y = data['Potability']

    # normalize the input values with a standard scalar 
    st = StandardScaler()
    col = x.columns
    x[col] = st.fit_transform(x[col])

    # split data into train and test splits
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # SMOTE (talked about more in report) generates data augmentation for our minority class
    # This will hopefully combat our class issue and generate more data for our model
    smote = SMOTE(random_state=42)


    X_train_res, Y_train_res = X_train, Y_train

    # apply SMOTE n=10 times to generate more resampling data
    number_of_applications = 10
    for _ in range(number_of_applications):  
        X_train_res, Y_train_res = smote.fit_resample(X_train_res, Y_train_res)

    # Save X_test and Y_test values with pickle for Test.py
    pickle.dump(X_test, open(X_initial, 'wb'))
    pickle.dump(Y_test, open(Y_initial, 'wb'))

    return X_train_res, X_test, Y_train_res, Y_test

# Function for preprocessing data with an "optimal" emphasis on reducing the size of our model
# we will further expand on this in Part III (details in report)
def preprocess_optimized_data():

    # same as above
    data = pd.read_csv('./data/water_potability.csv')
    data = data.dropna()

    # create correlation matrix for determining relevant features
    corr_matrix = data.corr()

    # plot correlation matrix to see which features are highly correlated 
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

    # using an (experimented with) threshold, determine the best features to keep
    threshold = 0.01 
    corr_with_target = corr_matrix['Potability'].abs().drop('Potability')
    relevant_features = corr_with_target[corr_with_target > threshold].index.tolist()

    # same as above
    notpotable = data[data['Potability'] == 0]
    potable = data[data['Potability'] == 1]
    df_minority_upsampled = resample(potable, replace=True, n_samples=3000)
    data = pd.concat([notpotable, df_minority_upsampled])
    data = shuffle(data, random_state=42)

    # extract relevant features from data
    x = data[relevant_features].copy()
    y = data['Potability']

    # apply standard scalar
    st = StandardScaler()
    x[relevant_features] = st.fit_transform(x[relevant_features])

    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # apply smote
    smote = SMOTE(random_state=42)
    X_train_res, Y_train_res = X_train, Y_train
    number_of_applications = 10
    for _ in range(number_of_applications): 
        X_train_res, Y_train_res = smote.fit_resample(X_train_res, Y_train_res)

    # Save X_test and Y_test values with pickle for Test.py
    pickle.dump(X_test, open(X_optimized, 'wb'))
    pickle.dump(Y_test, open(Y_optimized, 'wb'))

    return X_train_res, X_test, Y_train_res, Y_test

# Function for building and evaluating a model based on training and testing data
# the model filename we pass in says where to save the model using Pickle
def build_evaluate_model(model_filename, X_train_res, X_test, Y_train_res, Y_test):

    # define our KNN model - we used the default k=5 parameter (maybe optimize in part III)
    knn = KNeighborsClassifier(n_neighbors=5)

    # to find the most optimal parameters for our model, we use a grid search cross validation over a grid of values
    # we then build the model with the best combination of parmaeters
    para_knn = {'n_neighbors': np.arange(1, 50)}

    # make sure model saves training scores for displaying results
    model = GridSearchCV(knn, param_grid=para_knn, cv=5, return_train_score=True)

    # fit model
    model.fit(X_train_res, Y_train_res)

    # save model using pickle to the appropriate filename  
    pickle.dump(model, open(model_filename, 'wb'))

    # predict and compute accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    print("K-Nearest Neighbors Accuracy:", accuracy)
    print("K-Nearest Neighbors Model Size", os.path.getsize(model_filename))

    # get training scores from our model
    param_range = model.cv_results_['param_n_neighbors'].data  
    mean_train_score = model.cv_results_['mean_train_score']
    mean_test_score = model.cv_results_['mean_test_score']

    # ploting training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, mean_train_score, label="Training Accuracy", color='blue', marker='o')
    plt.plot(param_range, mean_test_score, label="Validation Accuracy", color='green', marker='s')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for KNN')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    # Initially preprocess data for first evaluation
    X_train_res, X_test, Y_train_res, Y_test = preprocess_initial_data()

    # Build, train, evaluate, and save the initial model
    print("Training initial model...")
    build_evaluate_model(initial_model_filename, X_train_res, X_test, Y_train_res, Y_test)

    # Optimally preprocess data 
    X_train_res, X_test, Y_train_res, Y_test = preprocess_optimized_data()

    # Build, train, evaluate, and save the optimized model
    print("Training optimal model...")
    build_evaluate_model(optimized_model_filename, X_train_res, X_test, Y_train_res, Y_test)



    
    


