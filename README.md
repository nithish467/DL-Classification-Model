# Developing a Neural Network Classification Model
### Name: NITHISHKUMAR S
### Register Number: 212223240109
## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1

Import required libraries such as PyTorch, NumPy, and Scikit-learn.

### STEP 2

Load the Iris dataset and split it into training and testing sets.

### STEP 3

Define the neural network architecture using nn.Module.

### STEP 4

Initialize the loss function (Cross-Entropy Loss) and optimizer (Adam).

### STEP 5

Train the model using forward propagation, loss calculation, and backpropagation.

### STEP 6

Evaluate the model using confusion matrix, classification report, and test predictions.



## PROGRAM


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (already numerical)



# Convert to DataFrame for easy inspection
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y


# Display first and last 5 rows
print("First 5 rows of dataset:\n", df.head())
print("\nLast 5 rows of dataset:\n", df.tail())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# Define Neural Network Model
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        #Include your code here
        self.fc1 =nn.Linear(input_size,16)
        self.fc2 =nn.Linear(16,8)
        self.fc3 =nn.Linear(8,3)



    def forward(self, x):
        #Include your code here
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
     #Include your code here
      for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# Initialize model, loss function, and optimizer
model =IrisClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)


# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=100)


# Evaluate the model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())


# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=iris.target_names)

# Print details
print("\nName: NITHISHKUMAR S")
print("Register No: 212223240109")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# Make a sample prediction
sample_input = X_test[5].unsqueeze(0)  # Removed unnecessary .clone()
with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = iris.target_names[predicted_class_index]

print("\nName: NITHISHKUMAR S")
print("Register No: 212223240109")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')


```

### Dataset Information
<img width="621" height="628" alt="image" src="https://github.com/user-attachments/assets/76971ea2-53d5-401c-b968-7d007469c809" />

### OUTPUT

## Confusion Matrix

<img width="703" height="126" alt="image" src="https://github.com/user-attachments/assets/59f2953c-88d0-40ad-9f3a-e0aa1b7690b2" />


<img width="478" height="445" alt="image" src="https://github.com/user-attachments/assets/69aa432e-1078-4898-916e-28ebbe59fe7c" />


## Classification Report

<img width="736" height="176" alt="image" src="https://github.com/user-attachments/assets/cbd35367-2dc9-4c06-a4d2-654e574c95b7" />


### New Sample Data Prediction
<img width="379" height="81" alt="image" src="https://github.com/user-attachments/assets/bfa11eb5-5ae4-460d-aa8d-e691a5b13309" />

## RESULT
Thus, a neural network classification model was successfully developed using the Iris dataset and its performance was evaluated using a confusion matrix and classification report.
