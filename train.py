import torch
import torch.distributed as dist
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
from improvements import ImprovedMrCNN


# Set up cuda
torch.backends.cudnn.enabled = True


def apply_conv1_weight_constraint(model, using_windows, world_size, weight_constraint=0.1):
    with torch.no_grad():
        # Get the first convolution layer in each stream
        if not using_windows and world_size > 1:
            convs = [model.module.stream1.conv1, model.module.stream2.conv1, model.module.stream3.conv1]
        else:
            convs = [model.stream1.conv1, model.stream2.conv1, model.stream3.conv1]
        
        # For each of the first convolution layers
        for conv in convs:
            # Calculate the l2 norm
            norm = torch.linalg.norm(conv.weight, ord=2, dim=(2,3), keepdim=True)

            # Find weights where the l2 norm exceeds the weight constraint
            norm_condition = (norm > weight_constraint).expand_as(conv.weight.data)

            # Normalise theese weights
            conv.weight.data[norm_condition] /= (norm + 1e-8).expand_as(conv.weight.data)[norm_condition]

def increase_momentum(optimizer, momentum_delta):
    # Increase the momentum by the linear momentum delta
    optimizer.param_groups[0]['momentum'] =optimizer.param_groups[0]['momentum'] + momentum_delta

def train_epoch(model, train_loader, optimizer, criterion, momentum_delta, conv1_weight_constraint, using_windows, world_size, device):
    # Set the model to training mode
    model.train()  

    running_loss = 0.0  
    correct_predictions = 0
    total_samples = 0

    # Iterate over the training data
    for _, (inputs, labels) in enumerate(train_loader):    
        
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

        # Linearly increase momentum to end momentum
        increase_momentum(optimizer, momentum_delta)
        
        # Apply weight constraint to the first conv layer in the network
        apply_conv1_weight_constraint(model, weight_constraint=conv1_weight_constraint, using_windows=using_windows, world_size=world_size)

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
                map_pixel_index = pixel_index % val_data.num_crops
                
                # Get index of fixation value
                row = map_pixel_index // 50
                col = map_pixel_index % 50

                # Write the fixation value to the necessary fixation map
                saliency_maps[map_index, col, row] = fixations[i].item()

                pixel_index += 1                

        preds = {}
        targets = {}

        for map_idx in range(saliency_maps.shape[0]):
            _, H, W = val_data.dataset[map_idx]["X"].cpu().numpy().shape
            image_name = val_data.dataset[map_idx]["file"].replace(".jpeg", "")

            # Convert the NumPy array to a PIL Image
            saliency_map = Image.fromarray(saliency_maps[map_idx].T)

            # Resize the image
            pred_fixMap = np.array( saliency_map.resize((W, H), resample=Image.BICUBIC) )

            # Obtain the ground truth fixation map
            GT_fixMap = Image.open(os.path.join(GT_fixations_dir, f"{image_name}_fixMap.jpg"))
            GT_fixMap = np.array(GT_fixMap)

            # Add to dictionary
            preds[image_name] = pred_fixMap
            targets[image_name] = GT_fixMap

        # Calculate validation auc metric
        avg_auc = calculate_auc(preds, targets)

        return avg_auc

def save(model, out_dir, train_metrics, epoch, hyperparameters):

    # Save trained model
    model_path = os.path.join(out_dir, f'cnn_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}.')

    # Write log about model and training performance
    save_log(out_dir, datetime.now(), **{**train_metrics, **hyperparameters})


def train(rank, 
          world_size, 
          data_dir,
          out_dir,
          use_val,
          total_epochs,
          start_epoch,
          batch_size,
          learning_rate,
          start_momentum,
          end_momentum,
          conv1_weight_constraint,
          weight_decay,
          early_stopping_patience,
          improvements,
          using_windows):
    
    
    multi_gpu = not using_windows and world_size > 1 # multi-gpu not supported on Windows
    verbose = ((rank == 0) or (not multi_gpu)) # if on main gpu or if using single gpu

    # setup the process groups if necessary
    if multi_gpu:
        setup_gpus(rank, world_size)

    # Get train and validation data
    if verbose: 
        print('Loading datasets...')
    train_data = MIT(dataset_path=os.path.join(data_dir, "train_data.pth.tar"))
    if use_val:
        val_data = MIT(dataset_path=os.path.join(data_dir, "val_data.pth.tar"))
    if verbose:
        print("Loaded datasets.")

    # Ground truth directory
    GT_fixations_dir = os.path.join(data_dir, "ALLFIXATIONMAPS")

    # Create data loaders
    if multi_gpu:
        train_loader = get_data_loader(train_data, rank, world_size, batch_size=batch_size)
        if use_val:
            val_loader = get_data_loader(val_data, rank, world_size, batch_size=batch_size)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size)
        if use_val:
            val_loader = DataLoader(val_data, batch_size=batch_size)

    # Output directory
    out_dir = os.path.join(out_dir, f'trained/{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')
    if verbose:
        os.makedirs(out_dir, exist_ok=True)

    # Create the model
    if improvements:
        model = ImprovedMrCNN().to(rank)
    else:
        model = MrCNN().to(rank)

    # Wrap model with DDP if necessary
    if multi_gpu:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Use cross entropy loss as stated in the reference paper
    criterion = nn.BCELoss().to(rank)

    # Initialise SGD optimiser
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=start_momentum, weight_decay=weight_decay)

    # Get the amount to increase the momentum by each iteration
    total_iterations = len(train_loader) * total_epochs
    momentum_delta = (end_momentum - start_momentum) / total_iterations # linear momentum increase

    if verbose:
        print(f'Starting training for {total_epochs} epochs ({total_iterations} iterations)')
    
    # For writing logs
    hyperparameters = {
        'batch_size' : batch_size,
        'learning_rate' : learning_rate,
        'start_momentum' : start_momentum,
        'end_momentum' : end_momentum,
        'momentum_delta' : momentum_delta,
        'lr_weight_decay' : weight_decay,
    }
    train_metrics = {
        "Average BCE loss per train epoch" : [],
        "Average accuracy per train epoch" : [],
        "Average val auc per train epoch" : [] 
    }

    # For early stopping
    best_avg_val_auc = 0
    epochs_since_checkpoint = 0

    train_start_time = time.time()

    # Training Loop
    for epoch in range(start_epoch, total_epochs):
        if multi_gpu:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch) 

        epoch_start_time = time.time()
        
        # Performing single train epoch and get train metrics
        avg_train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, momentum_delta, conv1_weight_constraint=conv1_weight_constraint, using_windows=using_windows, world_size=world_size, device=rank)

        if multi_gpu:
            # Average metrics over all gpus
            dist.barrier() 

            if rank == 0:
                avg_train_loss= torch.tensor(avg_train_loss).to(rank)
                torch.distributed.all_reduce(avg_train_loss, op=torch.distributed.ReduceOp.SUM)
                avg_train_loss /= world_size
                avg_train_loss = avg_train_loss.item()

                train_accuracy= torch.tensor(train_accuracy).to(rank)
                torch.distributed.all_reduce(train_accuracy, op=torch.distributed.ReduceOp.SUM)
                train_accuracy /= world_size
                train_accuracy = train_accuracy.item()

        # Log this epoch's training metrics
        train_metrics["Average BCE loss per train epoch"].append(round(avg_train_loss, 2))
        train_metrics["Average accuracy per train epoch"].append(round(train_accuracy, 2))

        # Do validation
        if use_val:
            # Perform single validation epoch and get validation metrics
            avg_val_auc = val_epoch(model, val_loader, val_data, GT_fixations_dir, device=rank)

            # Print epoch results with validation score
            epoch_time = time.strftime("%H:%M:%S", time.gmtime((time.time() - epoch_start_time)))
            if verbose:
                print(f"Epoch [{epoch+1}/{total_epochs}] (time: {epoch_time}), Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}, Validaton mean auc: {avg_val_auc}")

            
            if multi_gpu:
                # Get average validation auc over all gpus
                dist.barrier()

                if rank == 0:
                    avg_val_auc = torch.tensor(avg_val_auc).to(rank)
                    torch.distributed.all_reduce(avg_val_auc, op=torch.distributed.ReduceOp.SUM)
                    avg_val_auc /= world_size
                    avg_val_auc = avg_val_auc.item()

            # Log this epoch's validation metrics
            train_metrics['Average val auc per train epoch'].append(round(avg_val_auc, 2))

            # Handle early stopping if required and if on the main gpu and don't bother for the first 30 epochs
            if early_stopping_patience != -1 and verbose and epoch > 30:  

                if best_avg_val_auc < avg_val_auc:
                    # If current validation score is better then save the model and the log  
                    hyperparameters['total_epochs'] = epoch
                    train_metrics['Train runtime'] = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start_time))

                    save(model, out_dir, train_metrics, epoch, hyperparameters)

                    epochs_since_checkpoint = 0
                    best_avg_val_auc = avg_val_auc
                else:
                    # If current validation score is worse then don't save the model
                    epochs_since_checkpoint += 1

                    # If it has been too long without seeing improvement, stop early
                    if epochs_since_checkpoint >= early_stopping_patience:
                        if verbose:
                            print('Validation AUC not increased for {epochs_since_checkpoint} epochs - stopping early.')
                            break

        else:
            # Print epoch results without validation score
            epoch_time = time.strftime("%H:%M:%S", time.gmtime((time.time() - epoch_start_time)))
            if verbose:
                print(f"Epoch [{epoch+1}/{total_epochs}] (time: {epoch_time}), Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}")
    
    if multi_gpu:
        # Wait for each gpu to finish the current epoch before starting the next
        dist.barrier()
        if dist.is_initialized():
                dist.destroy_process_group()

    if verbose:
        print('Done training.')



if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset containing *_data.pth.tar", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for saving model/log (default to cwd)", required='.')
    parser.add_argument('--epochs', type=int, help="Number of epochs to train with", default=20)
    parser.add_argument('--num_gpus', type=int, help="Number of gpus to train with", default=2)
    parser.add_argument('--use_val', type=bool, help='Track validation metrics during training', default=False)
    parser.add_argument('--batch_size', type=int, help="Minibatch size", default=256)
    parser.add_argument('--learning_rate', type=float, help="Learning rate", default=0.002)
    parser.add_argument('--start_momentum', type=float, help="Optimiser start momentum", default=0.9)
    parser.add_argument('--end_momentum', type=float, help="Optimiser end momentum", default=0.99)
    parser.add_argument('--lr_weight_decay', type=float, help="Learning rate weight decay", default=2e-4)
    parser.add_argument('--conv1_weight_constraint', type=float, help="L2 norm constraint in the first conv layer", default=0.1)
    parser.add_argument('--using_windows', type=bool, help='Whether the script is being executed on a Windows machine (default=False)', default=False)
    parser.add_argument('--improvements', type=int, help='If to use improvements and what improvements to use. 0: none, 1: blur model', default=0)
    parser.add_argument('--early_stopping_patience', type=int, help='How many epochs to wait without validation improvement before early stopping (default=-1 no early stopping)', default=-1)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    use_val = args.use_val

    print(use_val)
    print('ji' if use_val else 'dog')

    # Hyperparameters
    total_epochs = args.epochs
    start_epoch = 0
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    start_momentum = args.start_momentum
    end_momentum = args.end_momentum
    weight_decay = args.lr_weight_decay
    conv1_weight_constraint = args.conv1_weight_constraint

    # Get improvements
    early_stopping_patience = args.early_stopping_patience
    improvements = args.improvements

    # Initialise gpus
    world_size = args.num_gpus 
    using_windows = args.using_windows

    if using_windows or world_size == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(device, 1,
              data_dir,
              out_dir,
              use_val,
              total_epochs,
              start_epoch,
              batch_size,
              learning_rate,
              start_momentum,
              end_momentum,
              conv1_weight_constraint,
              weight_decay,
              early_stopping_patience,
              improvements,
              True) # using windows
    else:
        mp.spawn(
            train,
            args=(world_size,
                  data_dir,
                  out_dir,
                  use_val,
                  total_epochs,
                  start_epoch,
                  batch_size,
                  learning_rate,
                  start_momentum,
                  end_momentum,
                  conv1_weight_constraint,
                  weight_decay,
                  early_stopping_patience,
                  improvements,
                  False), # using windows
            nprocs=world_size)
    
    