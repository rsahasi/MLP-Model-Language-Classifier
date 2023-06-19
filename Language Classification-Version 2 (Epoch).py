import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define file paths
spanish_file_path = os.path.expanduser("/Users/25ruhans/Desktop/spanish.txt")
english_file_path = os.path.expanduser("/Users/25ruhans/Desktop/english.txt")
german_file_path = os.path.expanduser("/Users/25ruhans/Desktop/german.txt")
spanish_test_file_path = os.path.expanduser("/Users/25ruhans/Desktop/SpanishTest.txt")
english_test_file_path = os.path.expanduser("/Users/25ruhans/Desktop/EnglishTest.txt")
german_test_file_path = os.path.expanduser("/Users/25ruhans/Desktop/GermanTest.txt")

# Load words from file
def load_words(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        words = [word.strip() for line in file for word in line.split() if len(word) == 5]
    return words

# Load training words
spanish_words = load_words(spanish_file_path)
english_words = load_words(english_file_path)
german_words = load_words(german_file_path)

# Convert training words to tensors
spanish_word_tensors = [torch.tensor([ord(char) for char in word]) for word in spanish_words]
english_word_tensors = [torch.tensor([ord(char) for char in word]) for word in english_words]
german_word_tensors = [torch.tensor([ord(char) for char in word]) for word in german_words]

# Determine the maximum length of word tensors
max_length = max(max(len(tensor) for tensor in spanish_word_tensors),
                 max(len(tensor) for tensor in english_word_tensors),
                 max(len(tensor) for tensor in german_word_tensors))

# Pad the word tensors to have the same length
pad_value = 0  
spanish_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in spanish_word_tensors]
english_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in english_word_tensors]
german_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in german_word_tensors]

training = spanish_word_tensors + english_word_tensors + german_word_tensors
training_data_2d = torch.stack(training)

# Define the target labels
spanish_size = len(spanish_word_tensors)
english_size = len(english_word_tensors)
german_size = len(german_word_tensors)
target = [2] * spanish_size + [0] * english_size + [1] * german_size

# Load test words
spanish_test_words = load_words(spanish_test_file_path)
english_test_words = load_words(english_test_file_path)
german_test_words = load_words(german_test_file_path)

# Convert test words to tensors
spanish_test_word_tensors = [torch.tensor([ord(char) for char in word]) for word in spanish_test_words]
english_test_word_tensors = [torch.tensor([ord(char) for char in word]) for word in english_test_words]
german_test_word_tensors = [torch.tensor([ord(char) for char in word]) for word in german_test_words]

spanish_test_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in spanish_test_word_tensors]
english_test_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in english_test_word_tensors]
german_test_word_tensors = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=pad_value) for tensor in german_test_word_tensors]

testing = spanish_test_word_tensors + english_test_word_tensors + german_test_word_tensors
testing_data_2d = torch.stack(testing)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x.float()))  # Convert input to float
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# hyperparameters
input_size = max_length
hidden_size = 5
num_classes = 3
learning_rate = 0.001
num_epochs = 50
batch_size = 16

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(training_data_2d, target, test_size=0.2, random_state=42)

# Convert training and test data to tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

mlp_model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate)

loss_values = []
accuracy_values = []

for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        outputs = mlp_model(inputs)
        
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_values.append(loss.item())

    mlp_model.eval()
    with torch.no_grad():
        test_predictions = mlp_model(X_test.float())
        test_predictions = torch.argmax(test_predictions, dim=1)
        test_correct = (test_predictions == y_test).sum().item()
        test_accuracy = test_correct / len(y_test)
        accuracy_values.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plotting loss vs. epoch
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Plotting accuracy vs. epoch
plt.plot(range(1, num_epochs + 1), accuracy_values)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MLP Accuracy vs. Epoch')
plt.show()

print(f"Final Accuracy: {accuracy_values[-1]:.4f}")
