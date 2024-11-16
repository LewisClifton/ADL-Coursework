import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
import numpy as np
import cv2

from dataset import MIT
from model import MrCNN

# Get train data
train_data = MIT(dataset_path="data/train_data.pth.tar")
train_data = Subset(train_data, list(range(5)))
train_loader = DataLoader(train_data, batch_size=1)

# Get validation data
val_data = MIT(dataset_path="data/val_data.pth.tar")
val_loader = DataLoader(val_data, batch_size=2)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 1e-4
num_epochs = 1

# Initialise model
model = MrCNN().to(device)

# Use cross entropy loss as stated in the reference paper
criterion = nn.BCELoss()

# Use adam optimiser
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()  

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # # Iterate over the training data
    # for (inputs, labels) in train_loader:    
        
    #     # Get the different resolutions for each image in the batch (shape=[batch_size, 3, 42, 42])
    #     x1 = inputs[:, 0, :, :, :].to(device) # 400x400
    #     x2 = inputs[:, 1, :, :, :].to(device) # 250x250
    #     x3 = inputs[:, 2, :, :, :].to(device) # 150x150

    #     labels = labels.to(device).float()
        
    #     # Zero the gradients
    #     optimizer.zero_grad()

    #     # Get model output for the batch
    #     outputs = model(x1, x2, x3).squeeze(1)

    #     # Compute the batch loss
    #     loss = criterion(outputs, labels)
        
    #     # Backward pass: Compute gradients
    #     loss.backward()
        
    #     # Update the weights
    #     optimizer.step()

    #     # Track the running loss
    #     running_loss += loss.item()

    #     # Convert predictions to binary (0 or 1) for accuracy calculation
    #     predicted = (outputs > 0.5).float()

    #     # Calculate number of correct predictions
    #     correct_predictions += (predicted.squeeze() == labels).sum().item()
    #     total_samples += labels.size(0)

    # # Calculate the average loss and accuracy for the current epoch
    # avg_loss = running_loss / len(train_loader)
    # accuracy = correct_predictions / total_samples * 100

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Validation step after each epoch
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_predictions_val = 0
    total_samples_val = 0
    print("TEST")
    with torch.no_grad():  # No gradient computation during validation
        for _, (inputs, labels)  in enumerate(val_loader):
            print("TEST")
            # Get the different resolutions for each image in the batch (shape=[batch_size, 3, 42, 42])
            x1 = inputs[:, 0, :, :, :].to(device) # 400x400
            x2 = inputs[:, 1, :, :, :].to(device) # 250x250
            x3 = inputs[:, 2, :, :, :].to(device) # 150x150

            labels = labels.to(device)

            # Forward pass on validation data
            outputs = model(x1, x2, x3)

            # Compute the validation loss
            loss = criterion(outputs.squeeze(), labels.float())
            
            val_loss += loss.item()

            # Convert predictions to binary for accuracy calculation
            predicted = (outputs > 0.5).float()
            
            # Calculate number of correct predictions
            correct_predictions_val += (predicted.squeeze() == labels).sum().item()
            total_samples_val += labels.size(0)

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_predictions_val / total_samples_val * 100

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Optionally save the model at intervals
    # torch.save(model.state_dict(), f'mrcnn_epoch_{epoch+1}.pth')

