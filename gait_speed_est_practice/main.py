from torch.utils.data import DataLoader
from utils import DatasetFromPath, segment_data
from Dataset import GaitDataset
from LSTMmodel import GaitLSTM
from sklearn.model_selection import train_test_split
import torch

# Directory of Dataset
raw_data_path = "IMU_data"
processed_data_path = "processed_data"


# Load data
# The dataset contains 4 IMU sensors, gait speed, subject, and condition information
# and has been processed (using StandardScaler) 
DataSet = DatasetFromPath(raw_data_path, processed_data_path)

print('DataSet.head() : \n', DataSet.head())

# Segmenting data
window_size = 200
stride = 50
X, y = segment_data(DataSet, window_size, stride)

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = GaitDataset(X_train, y_train)
test_dataset = GaitDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ML model parameter
input_size = X_train.shape[2]
hidden_size = 128
num_layers = 3
output_size = 1
dropout_prob = 0.4

# Instantiate the model, loss function, and optimizer
model = GaitLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

# Save the model's state_dict(parameter)
torch.save(model.state_dict(), 'LSTMmodel.pth')   



