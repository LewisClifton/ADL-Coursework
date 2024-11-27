import re
from typing import OrderedDict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


# Load model
def load_model(model, model_path):
    
    model_state_dict = torch.load(model_path, weights_only=True)

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in model_state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = model_state_dict

    model.load_state_dict(model_dict)

    return model


def evaluate(model, test_loader, test_data, GT_fixations_dir, image_dir, device, num_saved_images=5):
    # Set the model to evaluation mode
    model.eval()  

    # Create empty predicted saliency maps to populate using the model
    saliency_maps = np.empty( shape=(len(test_data.dataset), 50, 50) )

    with torch.no_grad(): 

        pixel_index = 0

        for _, (inputs, _) in enumerate(test_loader):
            
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
                map_index = pixel_index // test_data.num_crops
                map_pixel_index = pixel_index % test_data.num_crops
                
                # Get index of fixation value
                row = map_pixel_index // 50
                col = map_pixel_index % 50

                # Write the fixation value to the necessary fixation map
                saliency_maps[map_index, col, row] = fixations[i].item()

                pixel_index += 1                

        preds = {}
        targets = {}

        for map_idx in range(saliency_maps.shape[0]):
            _, H, W = test_data.dataset[map_idx]["X"].cpu().numpy().shape
            image_name = test_data.dataset[map_idx]["file"].replace(".jpeg", "")

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

            if map_idx < num_saved_images and device == 0:

                # Concatenate all three images along the width (axis 1) and convert to PIL image
                concatenated_image = np.concatenate((pred_fixMap * 255, GT_fixMap), axis=1)
                concatenated_image = Image.fromarray( (concatenated_image).astype(np.uint8) ) 
            
                # Save the concatenated image
                concatenated_image.save(os.path.join(image_dir, f"{image_name}_comparison.jpg"))

        # Calculate validation auc metric
        avg_auc = calculate_auc(preds, targets)

        return avg_auc
    

def main(rank, 
          world_size,
          data_dir,
          out_dir,
          model_path,
          batch_size,
          num_saved_images):

    # setup the process groups
    setup_gpus(rank, world_size)

    # Get train and validation data
    test_data = MIT(dataset_path=os.path.join(data_dir, "test_data.pth.tar"))

    # Ground truth directory
    GT_fixations_dir = os.path.join(data_dir, "ALLFIXATIONMAPS")

    # Create data loaders
    test_loader = get_data_loader(test_data, rank, world_size, batch_size=batch_size)

    # Load the trained model
    model = MrCNN().to(rank)
    model = load_model(model, model_path)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Output directory
    date = datetime.now()
    out_dir = os.path.join(out_dir, f'eval/{date.strftime("%Y_%m_%d_%p%I_%M")}')
    if not os.path.exists(out_dir) and rank == 0:
        os.makedirs(out_dir)

    # Directory to save images to
    image_dir = os.path.join(out_dir, "Comparison images")
    if not os.path.exists(image_dir) and rank == 0:
        os.makedirs(image_dir, exist_ok=True)

    # For gettin runtime
    eval_start_time = time.time()

    # Evaluate over the test set
    test_avg_auc = evaluate(model, test_loader, test_data, GT_fixations_dir, image_dir, num_saved_images=num_saved_images, device=rank)

    # Get runtime
    runtime = time.time() - eval_start_time

    # Wait for all the GPUs to finish evaluating
    dist.barrier()

    # Send all the gpu node metrics back to the main gpu
    test_metrics = {
        'Average test AUC' : test_avg_auc,
        'Test runtime' : runtime,
    }
    torch.cuda.set_device(rank)
    test_metrics_gpus = [None for _ in range(world_size)]
    dist.all_gather_object(test_metrics_gpus, test_metrics)

    if rank == 0:

        # Calculate average AUC over all the GPUs
        test_avg_aucs = [gpu_metrics['Average test AUC'] for gpu_metrics in test_metrics_gpus]
        test_avg_auc = np.mean(np.vstack(test_avg_aucs), axis=0)

        # Get runtime
        runtime = np.max([gpu_metrics['Test runtime'] for gpu_metrics in test_metrics_gpus])
        runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

        final_test_metrics = {
            'Model path' : model_path,
            'Average test AUC' : test_avg_auc,
            'Test runtime' : runtime,
        }

        # Save metrics
        save_log(out_dir, date, **{**final_test_metrics})

        if num_saved_images > 0:
            print(f'Saved {num_saved_images} to {image_dir}.')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for saving model/log/images", required=True)
    parser.add_argument('--model_path', type=str, help="Path of model to evaluate")
    parser.add_argument('--batch_size', type=int, help="Data loader batch size", default=256)
    parser.add_argument('--num_gpus', type=int, help="Number of gpus to train with", default=1)
    parser.add_argument('--num_saved_images', type=int, help="Number of comparison images to save", default=0)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    model_path = args.model_path
    batch_size = args.batch_size
    num_saved_images = args.num_saved_images

    # Initialise gpus
    world_size = args.num_gpus 
    mp.spawn(
        main,
        args=(world_size,
              data_dir,
              out_dir,
              model_path,
              batch_size,
              num_saved_images),
        nprocs=world_size)
    
    