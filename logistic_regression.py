#Import Libraries:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
import torch
import torch.nn as nn

# Load Dataset
data = pd.read_csv('./data/water_potability.csv')

# Print the first 5 rows of the dataframe
print(data.head())

# drop rows with missing values
data = data.dropna()

# Split the data into features and target
X = data.drop('Potability', axis=1)
Y = data['Potability']

# Split the data into training, Validation and Testing
X_train, X_temp, Y_train, Y_temp = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)
X_val, X_test, Y_val, Y_test = sklearn.model_selection.train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0)

# Normalize the data
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Apply PCA to reduce the dimensionality of the data
pca = sklearn.decomposition.PCA(n_components=9)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.int64)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val.values, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test.values, dtype=torch.int64)

# Create a DataLoader for the training data
train_data = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# Create a DataLoader for the validation data
val_data = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# Create a DataLoader for the testing data
test_data = torch.utils.data.TensorDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Define the Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(in_features=9, out_features=2)
        self.crossentropyloss = nn.CrossEntropyLoss()

    # Define the forward pass
    def forward(self, x):
        logits = self.fc(x)
        probabilties = torch.sigmoid(logits)
        return probabilties

    # Define the training method
    def train_model(self, train_loader, optimizer, device):
        self.train()
        overall_loss = 0
        
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.crossentropyloss(outputs, labels)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

        return overall_loss

    # Define the evaluation method
    def evaluate_model(self, val_loader, device):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy
    
# Initialize the model
model = LogisticRegression()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model.to(device)

# Train the model
num_epochs = 200
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss = model.train_model(train_loader, optimizer, device)
    val_accuracy = model.evaluate_model(val_loader, device)
    
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Evaluate the model on the test data
test_accuracy = model.evaluate_model(test_loader, device)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot the training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.show()


