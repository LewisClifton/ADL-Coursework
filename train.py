import re
from typing import OrderedDict
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import Image
from datetime import datetime
import time
import os
import argparse
import torch.multiprocessing as mp

from dataset import MIT
from model import MrCNN
from metrics import calculate_auc
from utils import *


# Set up cuda
torch.backends.cudnn.enabled = True

# Setup for multi-gpu loading
def setup_gpus(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Get the dataloaders
def get_data_loader(dataset, rank, world_size, batch_size=32):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

# Load model checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['model_state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = checkpoint['model_state_dict']

    model.load_state_dict(model_dict)
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
        for conv in [model.module.stream1.conv1, model.module.stream2.conv1, model.module.stream3.conv1]:
            norm = torch.linalg.norm(conv.weight, ord=2, dim=(2,3), keepdim=True)
            norm_condition = (norm > weight_constraint).expand_as(conv.weight.data)
            conv.weight.data[norm_condition] /= (norm + 1e-8).expand_as(conv.weight.data)[norm_condition]

def increase_momentum_linear(optimizer, start_momentum, momentum_delta, epoch):
    optimizer.param_groups[0]['momentum'] = start_momentum + epoch * momentum_delta

def train_epoch(model, train_loader, optimizer, criterion, device):
    # Set the model to training mode
    model.train()  

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Iterate over the training data
    for (inputs, labels) in train_loader:    
        
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


def val_epoch(model, val_loader, val_data, GT_fixations_dir, device):
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

            # Convert the NumPy array to a PIL Image
            saliency_map = Image.fromarray(saliency_maps[map_idx])

            # Resize the image
            pred_fixMap = np.array( saliency_map.resize((W, H), resample=Image.BICUBIC) )

            # Obtain the ground truth fixation map
            GT_fixMap = Image.open(os.path.join(GT_fixations_dir, f"{image_name}_fixMap.jpg"))
            GT_fixMap = np.array(GT_fixMap)

            # GT_fixPoints = Image.open(os.path.join(os.getcwd(), GT_fixations_dir, f"{name}_fixPts.jpg"))
            # GT_fixPoints = np.array(GT_fixPoints)

            # Add to dictionary
            preds[image_name] = pred_fixMap
            targets[image_name] = GT_fixMap

        # Calculate validation auc metric
        avg_auc = calculate_auc(preds, targets)

        return avg_auc


def train(rank, world_size):

    # setup the process groups
    setup_gpus(rank, world_size)

    # Get train and validation data
    train_data = MIT(dataset_path=os.path.join(data_dir, "train_data.pth.tar"))
    val_data = MIT(dataset_path=os.path.join(data_dir, "val_data.pth.tar"))
    print("Loaded datasets.")

    # Ground truth directory
    GT_fixations_dir = os.path.join(data_dir, "ALLFIXATIONMAPS")

    # Create data loaders
    train_loader = get_data_loader(train_data, rank, world_size, batch_size=batch_size)
    val_loader = get_data_loader(val_data, rank, world_size, batch_size=batch_size)

    # Create the model
    model = MrCNN().to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Use cross entropy loss as stated in the reference paper
    criterion = nn.BCELoss()

    # Initialise SGD optimiser
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=start_momentum, weight_decay=weight_decay)

    # Load from checkpoints if required
    if checkpoint_dir:
        print("Loading from checkpoint")
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_dir)

    print(f'Starting training (Epoch {start_epoch+1}/{total_epochs})')

    train_metrics = {
        "Average BCE loss per train epoch" : [],
        "Average accuracy per train epoch" : [],
        "Average val auc per train epoch" : [] 
    }

    train_start_time = time.time()

    # Training Loop
    for epoch in range(start_epoch, total_epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch) 

        epoch_start_time = time.time()
        
        # Performing single train epoch and get train metrics
        avg_train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device=rank)

        train_metrics["Average BCE loss per train epoch"].append(round(avg_train_loss, 2))
        train_metrics["Average accuracy per train epoch"].append(round(train_accuracy, 2))

        if use_val:
            # Perform single validation epoch and get validation metrics
            avg_val_auc = val_epoch(model, val_loader, val_data, GT_fixations_dir, device=rank)

            epoch_time = (time.time() - epoch_start_time).strftime("%H:%M:%S")
            print(f"Epoch [{epoch+1}/{total_epochs}] (time: {epoch_time}), Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}, Validaton mean auc: {avg_val_auc}")

            train_metrics['Average val auc per train epoch'].append(round(avg_train_loss, 2))
        else:
            epoch_time = (time.time() - epoch_start_time).strftime("%H:%M:%S")
            print(f"Epoch [{epoch+1}/{total_epochs}] (time: {epoch_time}), Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}")

        # Linearly increase momentum to end momentum
        increase_momentum_linear(optimizer, start_momentum, momentum_delta, epoch)
        
        # Apply weight constraint to the first conv layer in the network
        apply_conv_weight_constraint(model)

        # Save checkpoint
        if checkpoint_freq != -1:
            if epoch % checkpoint_freq == 0:
                save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    # Get runtime
    runtime = (time.time() - train_start_time).strftime("%H:%M:%S")

    # Send all the gpu node metrics back to the main gpu
    torch.cuda.set_device(rank)
    train_metrics_gpus = [None for _ in range(world_size)]
    dist.all_gather_object(train_metrics_gpus, train_metrics)

    if rank == 0:

        # Split the metrics from train_metrics_gpus
        bce_losses = [gpu_metrics["Average BCE loss per train epoch"] for gpu_metrics in train_metrics_gpus]
        avg_accs = [gpu_metrics["Average accuracy per train epoch"] for gpu_metrics in train_metrics_gpus]
        avg_val_aucs = [gpu_metrics["Average val auc per train epoch"] for gpu_metrics in train_metrics_gpus]

        # Stack the metrics and calculate mean across GPUs for each epoch
        bce_losses = np.mean(np.vstack(bce_losses), axis=0)
        avg_accs = np.mean(np.vstack(avg_accs), axis=0)
        avg_val_aucs = np.mean(np.vstack(avg_val_aucs), axis=0)

        # Final train metric for the log
        final_train_metrics = {
            "Average BCE loss per train epoch": bce_losses.tolist(),
            "Average accuracy per train epoch": avg_accs.tolist(),
            "Average val auc per train epoch": avg_val_aucs.tolist(),
        }

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
            'Time to train' : runtime,
        }
        save_log(out_dir, date, **{**final_train_metrics, **hyperparameters})


if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for saving model/log", required=True)
    parser.add_argument('--epochs', type=int, help="Number of epochs to train with", default=5)
    parser.add_argument('--num_gpus', type=int, help="Number of gpus to train with", default=2)
    parser.add_argument('--use_val', type=bool, help='Track validation metrics during training', default=False)
    parser.add_argument('--batch_size', type=int, help="Minibatch size", default=256)
    parser.add_argument('--learning_rate', type=float, help="Learning rate", default=0.02)
    parser.add_argument('--start_momentum', type=float, help="Optimiser start momentum", default=0.9)
    parser.add_argument('--end_momentum', type=float, help="Optimiser end momentum", default=0.99)
    parser.add_argument('--conv_weight_decay', type=float, help="Weight decay threshold of the first convolutional layer", default=2e-4)
    parser.add_argument('--checkpoint_path', type=str, help="Relative path of saved checkpoint from --out_dir")
    parser.add_argument('--checkpoint_freq', type=int, help="How many epochs between saving checkpoint. -1: don't save checkpoints", default=-1)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    use_val = args.use_val

    # Hyperparameters
    total_epochs = args.epochs
    start_epoch = 0
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    start_momentum = args.start_momentum
    end_momentum = args.end_momentum
    momentum_delta = (end_momentum - start_momentum) / total_epochs # linear momentum increase
    weight_decay = args.conv_weight_decay

    # Checkpoint path
    checkpoint_dir = args.checkpoint_path
    checkpoint_freq = args.checkpoint_freq # every checkpoint_freq epochs, save model checkpoint

    # Initialise gpus
    world_size = args.num_gpus 
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size)
    
    # Destroy processes
    dist.destroy_process_group()