import re
from typing import OrderedDict
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


# Set up cuda
torch.backends.cudnn.enabled = True

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

def apply_conv_weight_constraint(model, using_windows, weight_constraint=0.1):
    with torch.no_grad():
        if using_windows:
            convs = [model.stream1.conv1, model.stream2.conv1, model.stream3.conv1]
        else:
            convs = [model.module.stream1.conv1, model.module.stream2.conv1, model.module.stream3.conv1]
            
            
        for conv in convs:
            norm = torch.linalg.norm(conv.weight, ord=2, dim=(2,3), keepdim=True)
            norm_condition = (norm > weight_constraint).expand_as(conv.weight.data)
            conv.weight.data[norm_condition] /= (norm + 1e-8).expand_as(conv.weight.data)[norm_condition]

def increase_momentum(optimizer, momentum_delta):
    optimizer.param_groups[0]['momentum'] =optimizer.param_groups[0]['momentum'] + momentum_delta

def train_epoch(model, train_loader, optimizer, criterion, momentum_delta, using_windows, device):
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
        apply_conv_weight_constraint(model, using_windows=using_windows)

        break

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
          weight_decay,
          checkpoint_dir,
          checkpoint_freq,
          using_windows):

    # setup the process groups if necessary
    if not using_windows:
        setup_gpus(rank, world_size)

    # Get train and validation data
    train_data = MIT(dataset_path=os.path.join(data_dir, "train_data.pth.tar"))
    val_data = MIT(dataset_path=os.path.join(data_dir, "val_data.pth.tar"))
    if rank == 0:
        print("Loaded datasets.")

    # Ground truth directory
    GT_fixations_dir = os.path.join(data_dir, "ALLFIXATIONMAPS")

    # Create data loaders
    train_loader = get_data_loader(train_data, rank, world_size, batch_size=batch_size)
    val_loader = get_data_loader(val_data, rank, world_size, batch_size=batch_size)

    # Create the model
    model = MrCNN().to(rank)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Use cross entropy loss as stated in the reference paper
    criterion = nn.BCELoss().to(rank)

    # Initialise SGD optimiser
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=start_momentum, weight_decay=weight_decay)

    # Load from checkpoints if required
    if checkpoint_dir:
        if rank == 0:
            print("Loading from checkpoint")
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_dir)

    if rank == 0:
        print(f'Starting training (Epoch {start_epoch+1}/{total_epochs})')

    train_metrics = {
        "Average BCE loss per train epoch" : [],
        "Average accuracy per train epoch" : [],
        "Average val auc per train epoch" : [] 
    }

    # Get the amount to increase the momentum by each iteration
    total_iterations = len(train_loader) * total_epochs
    momentum_delta = (end_momentum - start_momentum) / total_iterations # linear momentum increase

    train_start_time = time.time()

    # Training Loop
    for epoch in range(start_epoch, total_epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch) 

        epoch_start_time = time.time()
        
        # Performing single train epoch and get train metrics
        avg_train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, momentum_delta, using_windows, device=rank)

        train_metrics["Average BCE loss per train epoch"].append(round(avg_train_loss, 2))
        train_metrics["Average accuracy per train epoch"].append(round(train_accuracy, 2))

        if use_val:
            # Perform single validation epoch and get validation metrics
            avg_val_auc = val_epoch(model, val_loader, val_data, GT_fixations_dir, device=rank)

            epoch_time = time.strftime("%H:%M:%S", time.gmtime((time.time() - epoch_start_time)))
            if rank == 0:
                print(f"Epoch [{epoch+1}/{total_epochs}] (time: {epoch_time}), Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}, Validaton mean auc: {avg_val_auc}")

            train_metrics['Average val auc per train epoch'].append(round(avg_val_auc, 2))
        else:
            epoch_time = time.strftime("%H:%M:%S", time.gmtime((time.time() - epoch_start_time)))
            if rank == 0:
                print(f"Epoch [{epoch+1}/{total_epochs}] (time: {epoch_time}), Train BCE loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.2f}")

        # Save checkpoint
        if rank == 0:
            if checkpoint_freq != -1:
                if epoch % checkpoint_freq == 0:
                    save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    # Get runtime
    train_metrics['Train runtime'] = time.time() - train_start_time

    if not using_windows:
        dist.barrier()

        # Send all the gpu node metrics back to the main gpu
        torch.cuda.set_device(rank)
        train_metrics_gpus = [None for _ in range(world_size)]
        dist.all_gather_object(train_metrics_gpus, train_metrics)

        if rank == 0:
            print('Done training')

            # Get runtime
            runtime = np.max([gpu_metrics['Train runtime'] for gpu_metrics in train_metrics_gpus])
            runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

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
                "Train runtime" : runtime,
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
            model_path = os.path.join(out_dir, f'cnn.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}.')

            # Write log about model and training performance
            hyperparameters = {
                'total_epochs' : total_epochs,
                'start_epoch' : start_epoch,
                'batch_size' : batch_size,
                'learning_rate' : learning_rate,
                'start_momentum' : start_momentum,
                'end_momentum' : end_momentum,
                'momentum_delta' : momentum_delta,
                'lr_weight_decay' : weight_decay,
            }
            save_log(out_dir, date, **{**final_train_metrics, **hyperparameters})

        if dist.is_initialized():
            dist.destroy_process_group()

    else: # using single gpu
        # Final train metric for the log
            train_metrics["Train runtime"] = time.strftime("%H:%M:%S", time.gmtime(train_metrics["Train runtime"]))

            # Output directory
            date = datetime.now()
            out_dir = os.path.join(out_dir, f'trained/{date.strftime("%Y_%m_%d_%p%I_%M")}')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Save trained model
            model_path = os.path.join(out_dir, f'cnn.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}.')

            # Write log about model and training performance
            hyperparameters = {
                'total_epochs' : total_epochs,
                'start_epoch' : start_epoch,
                'batch_size' : batch_size,
                'learning_rate' : learning_rate,
                'start_momentum' : start_momentum,
                'end_momentum' : end_momentum,
                'momentum_delta' : momentum_delta,
                'lr_weight_decay' : weight_decay,
            }
            save_log(out_dir, date, **{**train_metrics, **hyperparameters})



if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset containing *_data.pth.tar", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for saving model/log", required=True)
    parser.add_argument('--epochs', type=int, help="Number of epochs to train with", default=20)
    parser.add_argument('--num_gpus', type=int, help="Number of gpus to train with", default=2)
    parser.add_argument('--use_val', type=bool, help='Track validation metrics during training', default=False)
    parser.add_argument('--batch_size', type=int, help="Minibatch size", default=256)
    parser.add_argument('--learning_rate', type=float, help="Learning rate", default=0.002)
    parser.add_argument('--start_momentum', type=float, help="Optimiser start momentum", default=0.9)
    parser.add_argument('--end_momentum', type=float, help="Optimiser end momentum", default=0.99)
    parser.add_argument('--lr_weight_decay', type=float, help="Learning rate weight decay", default=2e-4)
    parser.add_argument('--checkpoint_path', type=str, help="Relative path of saved checkpoint from --out_dir")
    parser.add_argument('--checkpoint_freq', type=int, help="How many epochs between saving checkpoint. -1: don't save checkpoints", default=-1)
    parser.add_argument('--using_windows', type=bool, help='Whether the script is being executed on a Windows machine (default=False)', default=False)
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
    weight_decay = args.lr_weight_decay

    # Checkpoint path
    checkpoint_dir = args.checkpoint_path
    checkpoint_freq = args.checkpoint_freq # every checkpoint_freq epochs, save model checkpoint

    # Initialise gpus
    world_size = args.num_gpus 
    using_windows = args.using_windows

    if using_windows:
        train(0, 1,
              data_dir,
              out_dir,
              use_val,
              total_epochs,
              start_epoch,
              batch_size,
              learning_rate,
              start_momentum,
              end_momentum,
              weight_decay,
              checkpoint_dir,
              checkpoint_freq,
              True)
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
                  weight_decay,
                  checkpoint_dir,
                  checkpoint_freq,
                  False),
            nprocs=world_size)
    
    