import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import os
import argparse

from dataset import MIT
from model import MrCNN
from metrics import calculate_auc
from utils import *


# Set up cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True


# Load model checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    return model, optimizer, start_epoch


# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch},
                f"{checkpoint_dir}/{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}.pth")

def apply_conv_weight_constraint(model, weight_constraint=0.1):
    with torch.no_grad():
        for conv in [model.stream1.conv1, model.stream2.conv1, model.stream3.conv1]:
            norm = torch.linalg.norm(conv.weight, ord=2, dim=(2,3), keepdim=True)
            norm_condition = (norm > weight_constraint).expand_as(conv.weight.data)
            conv.weight.data[norm_condition] /= (norm + 1e-8).expand_as(conv.weight.data)[norm_condition]

def increase_momentum_linear(optimizer, start_momentum, momentum_delta, epoch):
    optimizer.param_groups[0]['momentum'] = start_momentum + epoch * momentum_delta

def train_epoch(model, train_loader, optimizer):
    # Set the model to training mode
    model.train()  

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch = 0

    # Iterate over the training data
    for (inputs, labels) in train_loader:    
        batch+=1
        print(batch)
        
        # Get the different resolutions for each image in the batch (shape=[batch_size, 3, 42, 42])
        x1 = inputs[:, 0, :, :, :].to(device) # 400x400
        x2 = inputs[:, 1, :, :, :].to(device) # 250x250
        x3 = inputs[:, 2, :, :, :].to(device) # 150x150

        labels = labels.to(device).float()
        
        # Zero the gradients
        optimizer.zero_grad()

        # Get model output for the batch
        outputs = model(x1, x2, x3).squeeze(1)

        # Compute the batch loss
        loss = criterion(outputs, labels)
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update the weights
        optimizer.step()

        # Track the running loss
        running_loss += loss.item()

        # Convert predictions to binary (0 or 1) for accuracy calculation
        predicted = (outputs > 0.5).float()

        # Calculate number of correct predictions
        correct_predictions += (predicted.squeeze() == labels).sum().item()
        total_samples += labels.size(0)

    # # Calculate the average loss and accuracy for the current epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples * 100

    return avg_loss, accuracy


def val_epoch(model, val_loader, val_data):
    # Set the model to evaluation mode
    model.eval()  

    # Create empty predicted saliency maps to populate using the model
    saliency_maps = np.empty( shape=(len(val_data.dataset), 50, 50) )

    with torch.no_grad(): 

        pixel_index = 0

        for _, (inputs, _) in enumerate(val_loader):
            
            # Get the different resolutions for each crop in the batch
            x1 = inputs[:, 0, :, :, :].to(device) # 400x400
            x2 = inputs[:, 1, :, :, :].to(device) # 250x250
            x3 = inputs[:, 2, :, :, :].to(device) # 150x150

            # Get fixation value for each crop in the batch
            fixations = model(x1, x2, x3).cpu().numpy()

            # Get the number of crop fixations predicted (batch size)
            pixels = fixations.shape[0]
            for i in range(pixels):

                # Get the fixation map this fixation value belonds to
                map_index = pixel_index // val_data.num_crops
                
                # Get index of fixation value
                row = (pixel_index - (map_index * val_data.num_crops)) // 50
                col = (pixel_index - (map_index * val_data.num_crops)) % 50

                # Write the fixation value to the necessary fixation map
                saliency_maps[map_index, row, col] = fixations[i].item()

                pixel_index += 1                

        preds = {}
        targets = {}

        for map_idx in range(saliency_maps.shape[0]):
            _, H, W = val_data.dataset[map_idx]["X"].cpu().numpy().shape
            image_name = val_data.dataset[map_idx]["file"].replace(".jpeg", "")

            # Upscale the predicted fixation map
            pred_fixMap = cv2.resize(saliency_maps[map_idx], (W, H), interpolation=cv2.INTER_CUBIC)

            # Obtain the ground truth fixation map
            GT_fixMap = Image.open(os.path.join(os.getcwd(), GT_fixations_dir, f"{image_name}_fixMap.jpg"))
            GT_fixMap = np.array(GT_fixMap)

            # GT_fixPoints = Image.open(os.path.join(os.getcwd(), GT_fixations_dir, f"{name}_fixPts.jpg"))
            # GT_fixPoints = np.array(GT_fixPoints)

            # Add to dictionary
            preds[image_name] = pred_fixMap
            targets[image_name] = GT_fixMap

        # Calculate validation auc metric
        avg_auc = calculate_auc(preds, targets)

        return avg_auc



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    
    out_dir = parser.out_dir

    # Checkpoint path
    checkpoint_dir = '' # os.path.join(os.getcwd(), 'checkpoints')
    load_from_checkpoint = False
    checkpoint_freq = 200 # every checkpoint_freq epochs, save model checkpoint

    # Get train and validation data
    train_data = MIT(dataset_path=os.path.join("data/train_data.pth.tar"))
    val_data = MIT(dataset_path=os.path.join("data/val_data.pth.tar"))
    print("Loaded datasets.")

    # Ground truth directory
    GT_fixations_dir = "data/ALLFIXATIONMAPS"

    # Hyperparameters
    total_epochs = 1
    start_epoch = 0
    batch_size = 256
    learning_rate = 0.02
    start_momentum = 0.9
    end_momentum = 0.99
    momentum_delta = (end_momentum - start_momentum) / total_epochs # linear momentum increase
    weight_decay = 2e-4
    
    # Initialise model
    model = MrCNN().to(device)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=256)
    val_loader = DataLoader(val_data, batch_size=256)

    # Use cross entropy loss as stated in the reference paper
    criterion = nn.BCELoss()

    # Initialise SGD optimiser
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=start_momentum, weight_decay=weight_decay)

    # Load from checkpoints if required
    if load_from_checkpoint:
        print("Loading from checkpoint")
        model, optimizer, start_epoch = load_checkpoint(checkpoint_dir)

    print(f'Starting training (Epoch {start_epoch+1}/{total_epochs})')

    train_metrics = {
        "Average BCE loss per train epoch" : [],
        "Average accuracy per train epoch" : [],
        "Average val auc per train epoch" : [] 
    }

    # Training Loop
    for epoch in range(start_epoch, total_epochs):
        
        # Performing single train epoch and get train metrics
        avg_train_loss, train_accuracy = train_epoch(model, train_loader, optimizer)

        # Perform single validation epoch and get validation metrics
        avg_val_auc = val_epoch(model, val_loader, val_data)

        train_metrics["Average BCE loss per train epoch"].append(avg_train_loss, 2)
        train_metrics["Average accuracy per train epoch"].append(train_accuracy, 2)
        train_metrics['Average val auc per train epoch'].append(avg_train_loss, 2)

        print(f"Epoch [{epoch+1}/{total_epochs}], Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}, Validaton mean auc: {avg_val_auc}")

        # Linearly increase momentum to end momentum
        increase_momentum_linear(optimizer, start_momentum, momentum_delta, epoch)
        
        # Apply weight constraint to the first conv layer in the network
        apply_conv_weight_constraint(model)

        # Save checkpoint
        if epoch % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    # Output directory
    date = datetime.now()
    out_dir = os.path.join(out_dir, f'trained/{date.strftime("%Y_%m_%d_%p%I_%M")}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save trained model
    torch.save(model.state_dict(), os.path.join(out_dir, f'cnn.pth'))

    # Write log about model and training performance
    hyperparameters = {
        'total_epochs' : total_epochs,
        'start_epoch' : start_epoch,
        'batch_size' : batch_size,
        'learning_rate' : learning_rate,
        'start_momentum' : start_momentum,
        'end_momentum' : end_momentum,
        'momentum_delta' : momentum_delta,
        'weight_decay' : weight_decay,
    }
    save_log(out_dir, date, **{**train_metrics, **hyperparameters})