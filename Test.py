# generally used
from mimetypes import init
from venv import create
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


# constants for where we save our initial model and our optimized model
initial_model_filename = "models/model.sav"
optimized_model_filename = "models/optimized_model.sav"

# constants for our saved test data files
X_initial = "data/X_initial.pkl"
Y_initial = "data/Y_initial.pkl"
X_optimized = "data/X_optimized.pkl"
Y_optimized = "data/Y_optimized.pkl"

def load_test_values(filename):

    if filename == initial_model_filename:
        loaded_X = pickle.load(open(X_initial, 'rb'))
        loaded_Y = pickle.load(open(Y_initial, 'rb'))

        return loaded_X, loaded_Y

    elif filename == optimized_model_filename:
        loaded_X = pickle.load(open(X_optimized, 'rb'))
        loaded_Y = pickle.load(open(Y_optimized, 'rb'))

        return loaded_X, loaded_Y


# Function for loading and evaluating a model
def load_evaluate_model(model_filename, X_test, Y_test):

    # load the model using Pickle
    loaded_model = pickle.load(open(model_filename, 'rb'))

    # predict and compute accuracy
    y_pred = loaded_model.predict(X_test)

    Y_test = Y_test.astype(int)
    y_pred = y_pred.astype(int)

    accuracy = accuracy_score(Y_test, y_pred)

    print("Loaded K-Nearest Neighbors Accuracy:", accuracy)
    print("Loaded K-Nearest Neighbors Model Size", os.path.getsize(model_filename))


if __name__ == "__main__":

    X_test, Y_test = load_test_values(initial_model_filename)

    # Load initial saved model (to avoid retraining everytime)
    load_evaluate_model(initial_model_filename, X_test, Y_test)

    X_test, Y_test = load_test_values(optimized_model_filename)

    # Load optimized saved model 
    load_evaluate_model(optimized_model_filename, X_test, Y_test)