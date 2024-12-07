# Import Libraries:
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

# Drop rows with missing values
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
#pca = sklearn.decomposition.PCA(n_components=9)
#X_train = pca.fit_transform(X_train)
#X_val = pca.transform(X_val)
#X_test = pca.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.int64)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val.values, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test.values, dtype=torch.int64)

# Create a DataLoader for the training data
train_data = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Create a DataLoader for the validation data
val_data = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

# Create a DataLoader for the testing data
test_data = torch.utils.data.TensorDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

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
        overall_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                loss = self.crossentropyloss(outputs, labels)
                overall_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = overall_loss / len(val_loader)
        return accuracy, avg_loss

# Function to experiment with different sets of features
def experiment_with_features(X_train, Y_train, X_val, Y_val, feature_sets):
    results = {}
    for feature_set in feature_sets:
        print(f"Evaluating feature set: {feature_set}")
        
        # Select the features
        X_train_subset = X_train[:, feature_set]
        X_val_subset = X_val[:, feature_set]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_subset, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_subset, dtype=torch.float32)
        
        # Create DataLoaders
        train_data = torch.utils.data.TensorDataset(X_train_tensor, Y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        
        val_data = torch.utils.data.TensorDataset(X_val_tensor, Y_val)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
        
        # Initialize the model
        model = LogisticRegression()
        model.fc = nn.Linear(in_features=len(feature_set), out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Train the model
        num_epochs = 40
        for epoch in range(num_epochs):
            model.train_model(train_loader, optimizer, device)
        
        # Evaluate the model
        val_accuracy, _ = model.evaluate_model(val_loader, device)
        results[tuple(feature_set)] = val_accuracy
        print(f"Validation Accuracy for feature set {feature_set}: {val_accuracy:.4f}")
    
    return results

# Define feature sets to experiment with
feature_sets = [
    #[0, 1, 2],  # First three features
    #[3, 4, 5],  # Next three features
    #[6, 7, 8],  # Last three features
    [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All features
]

# Run the experiments
results = experiment_with_features(X_train.numpy(), Y_train, X_val.numpy(), Y_val, feature_sets)
print("Experiment results:", results)

# Initialize the model
model = LogisticRegression()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model.to(device)

# Train the model
num_epochs = 100
train_losses = []
val_accuracies = []
val_losses = []

# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Update training loop
for epoch in range(num_epochs):
    train_loss = model.train_model(train_loader, optimizer, device)
    val_accuracy, val_loss = model.evaluate_model(val_loader, device)
    
    # Step the scheduler
    scheduler.step()
    
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}')

# Evaluate the model on the test data
test_accuracy, test_loss = model.evaluate_model(test_loader, device)
print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')

# Plot the training loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs Epoch')
plt.legend()
plt.show()


