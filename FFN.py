import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import mean_squared_error
import re

def parse_log(log_input):
    """
       Parse log content to extract relevant information about solver results.

       Args:
       log_input (str or file path): Content of the log or path to the log file.

       Returns:
       dict: Dictionary containing parsed information about solver results.
    """
    if isinstance(log_input, str):
        # If log_input is a string, treat it as the log content
        log_content = log_input
    elif os.path.isfile(log_input):
        # If log_input is a file path, read the content from the file
        with open(log_input, 'r') as file:
            log_content = file.read()
    else:
        raise ValueError("Invalid input. Please provide a valid file path or log content as a string.")

    # The rest of the function remains unchanged
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    id_pattern = re.compile(r'Solver Result: \(([^\)]+)')
    flips_var_pattern = re.compile(r'flips_var\': (\d+\.\d+)')
    decay_factor_var_pattern = re.compile(r'decay_factor_var\': (\d+\.\d+)')
    flips_pattern = re.compile(r'flips\': (\d+)')
    success_time_pattern = re.compile(r'succeeded in (\d+\.\d+) seconds\.')
    good_result_pattern = re.compile(r'Good result: (\w+)')

    timestamp_match = re.search(timestamp_pattern, log_content)
    if timestamp_match:
        solver_id = timestamp_match.group(1)
    else:
        solver_id = 'default_id'

    flips_var = float(re.search(flips_var_pattern, log_content).group(1))
    decay_factor_var = float(re.search(decay_factor_var_pattern, log_content).group(1))
    flips = int(re.search(flips_pattern, log_content).group(1))

    success_time_match = re.search(success_time_pattern, log_content)
    running_time = float(success_time_match.group(1)) if success_time_match else None

    good_result_str = re.search(good_result_pattern, log_content)

    good_result = True if good_result_str and 'Good' in good_result_str.group() else False
    time_str = re.search(timestamp_pattern, log_content).group(1)

    result_data = {
        'solver_id': solver_id,
        'flips_var': flips_var,
        'decay_factor_var': decay_factor_var,
        'flips': flips,
        'good_result': good_result,
        'running_time': running_time,
    }

    return result_data

def add_success_score(result_dict, good_result_weight=0.66, running_time_weight=0.165, flips_weight=0.165):
    """
       Calculate success score based on specified weights and update result dictionary.

       Args:
       result_dict (dict): Dictionary containing solver result information.
       good_result_weight (float): Weight assigned to good results.
       running_time_weight (float): Weight assigned to running time.
       flips_weight (float): Weight assigned to flips.

       Returns:
       dict: Updated result dictionary with success score.
    """
    max_running_time = 10.0
    max_flips = 50000

    good_result = result_dict['good_result']
    running_time = result_dict['running_time']
    flips = result_dict['flips']

    success_score = (
        good_result_weight * (1 if good_result else 0) +
        running_time_weight * ((max_running_time - running_time)/max_running_time) +
        flips_weight * ((max_flips - flips)/max_flips)
    )

    result_dict['success_score'] = success_score

    return result_dict


def process_log_file(log_path, good_result_weight=0.66, running_time_weight=0.165, flips_weight=0.165):
    """
       Process log file to extract and calculate information about solver results.

       Args:
       log_path (str): Path to the log file.
       good_result_weight (float): Weight assigned to good results.
       running_time_weight (float): Weight assigned to running time.
       flips_weight (float): Weight assigned to flips.

       Returns:
       list: List of dictionaries containing processed information about solver results.
       """
    with open(log_path, 'r') as file:
        log_content = file.read()

    # Split the log content into individual entries
    log_entries = log_content.split('\n')

    # Initialize an empty list to store the results
    solver_results_list = []

    # Process each entry in the log file
    for i in range(0, len(log_entries)-1, 5):
        # Call parse_log for each entry
        entry_string = '\n'.join(log_entries[i:i + 5])

        # Call parse_log for each entry
        result_dict = parse_log(entry_string)

        # Call add_success_score for each entry
        add_success_score(result_dict, good_result_weight, running_time_weight, flips_weight)

        # Append the result_dict to the list
        solver_results_list.append(result_dict)
    return solver_results_list

log_path = 'A:\projects for python\SAT Solver + FFN\solver_test_log.txt'
final_result = process_log_file(log_path)
for res in final_result:
    print("{\n")
    for r in res:
        print(r + " " + str(res[r]))
    print("\n}")

print(len(final_result))

data = process_log_file(log_path)
features = np.array([[d["flips_var"], d["decay_factor_var"], d["flips"]] for d in data])
labels = np.array([d["success_score"] for d in data])

features_tensor = torch.FloatTensor(features)
labels_tensor = torch.FloatTensor(labels).view(880, 1)  # Reshape to (num_samples, 1)

X_train, X_test, y_train, y_test = train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class FFN(nn.Module):
    def __init__(self, input_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Input size to 64 neurons in hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # 64 neurons in hidden layer to 1 output neuron

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_size = X_train.shape[1]  # Number of features
model = FFN(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
best_score = float('inf')
best_flips_var = None
best_decay_factor_var = None
best_flips = None
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

    # Evaluate the model
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    with torch.no_grad():
        for inputs, labels in test_loader:
            predicted_labels = model(inputs)
            mse = mean_squared_error(labels.numpy(), predicted_labels.numpy())
            if mse < best_score:
                best_score = mse
                best_flips_var, best_decay_factor_var, best_flips = inputs[torch.argmin(torch.abs(predicted_labels - labels))].tolist()

# Print the best features
print("Best Features:")
print(f"Flips Var: {best_flips_var}")
print(f"Decay Factor Var: {best_decay_factor_var}")
print(f"Flips: {best_flips}")