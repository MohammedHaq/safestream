# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample, shuffle
from imblearn.over_sampling import SMOTE  # For data augmentation using SMOTE

# Load data
data = pd.read_csv('./data/water_potability.csv')

# Display data info
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Balance the dataset
notpotable = data[data['Potability'] == 0]
potable = data[data['Potability'] == 1]
df_minority_upsampled = resample(potable, replace=True, n_samples=len(notpotable), random_state=42)
data = pd.concat([notpotable, df_minority_upsampled])
data = shuffle(data, random_state=42)

# Display balanced data info
print(data.Potability.value_counts())

# Print correlation with 'Potability'
print(data.corr()["Potability"].sort_values(ascending=False))

# Apply ML algorithms
x = data.drop(['Potability'], axis=1)
y = data['Potability']

# Ensure labels are integers for XGBoost compatibility
le = LabelEncoder()
y = le.fit_transform(y)

# Applying StandardScaler to normalize the features
st = StandardScaler()
col = x.columns
x[col] = st.fit_transform(x[col])

# Check for NaNs or invalid values
if np.any(np.isnan(x)) or np.any(np.isinf(x)):
    print("Warning: Data contains NaN or infinite values after scaling!")
    x = np.nan_to_num(x, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Apply data augmentation (SMOTE) multiple times to the training set
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = X_train, Y_train

for _ in range(5):  # Apply SMOTE 5 times to generate more data
    X_train_res, Y_train_res = smote.fit_resample(X_train_res, Y_train_res)

# Print the number of samples after augmentation
print(f"Original training set size: {len(X_train)}")
print(f"Augmented training set size: {len(X_train_res)}")

# Define models
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# Define parameter grids
para_knn = {'n_neighbors': np.arange(1, 50)}
grid_knn = GridSearchCV(knn, param_grid=para_knn, cv=5)

params_rf = {'n_estimators': [100, 200, 350, 500], 'min_samples_leaf': [2, 10, 30]}
grid_rf = GridSearchCV(rf, param_grid=params_rf, cv=5)

params_xgb = {'n_estimators': [50, 100, 250, 400, 600], 'learning_rate': [0.1, 0.2, 0.5, 0.8, 1]}
rs_xgb = RandomizedSearchCV(xgb, param_distributions=params_xgb, cv=5)

# Fit models
grid_knn.fit(X_train_res, Y_train_res)
grid_rf.fit(X_train_res, Y_train_res)
rs_xgb.fit(X_train_res, Y_train_res)

# Print best parameters (convert np.int64 to int for cleaner output)
print("Best parameters for KNN:", {k: int(v) if isinstance(v, np.int64) else v for k, v in grid_knn.best_params_.items()})
print("Best parameters for Random Forest:", {k: int(v) if isinstance(v, np.int64) else v for k, v in grid_rf.best_params_.items()})
print("Best parameters for XGBoost:", rs_xgb.best_params_)

# Evaluate models
models = [('K Nearest Neighbours', grid_knn.best_estimator_),
          ('Random Forest', grid_rf.best_estimator_),
          ('XGBoost', rs_xgb.best_estimator_)]

best_model_name = ""
best_accuracy = 0

for model_name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print('{:s} : {:.2f}'.format(model_name, accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name

print(f"Finally the best Model we can apply is {best_model_name} with accuracy {best_accuracy:.2f}.")