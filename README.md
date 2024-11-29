# An Improved Multi-Resolution CNN for Predicting Visual Saliency by Lewis Clifton

This repository contains the code for my implementation and improvement of the Mr-CNN saliency prediction model found at:

https://openaccess.thecvf.com/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf


## How to run

The scripts in this repo assumes it is being run using the lab machines. There is multi-gpu support for BC4 but it doesn't support validation metrics during training because Distributed Data Parallel made it tricky to synchronise metrics for creating model checkpoints (i.e. if running `train.py`, set `--use_val=0`). I highly advise running on the lab machine.

1. Create the conda environment with necessary packages (Pytorch (cuda), Pillow, Scipy, Numpy)
   If using the lab machine use the conda env for COMSM0159 as the Applied Deep Learning one has the wrong version of Scipy. 

## How to run train.py
Basic usage:
   ```
   python train.py --data_dir <DATA_DIR> --out_dir <OUT_DIR>
   ```

   Replace <DATA_DIR> with the path to the directory containing the *.pth.tar dataset files as well as the ALLFIXATIONMAPS folder provided in the dataset (if needing evaluation metrics during training) and <OUT_DIR> with the path to the directory where you want the trained model and log to be saved.

### Full train.py flags list

The default values of the non required flags are the hyperparameters used to train the base model.
The full list of flags can be given using `python train.py -h` but is also given below:

- `--data_dir`: Path to the dataset directory containing train_data.pth.tar, val_data.pth.tar (if --use_val=1)(required).
- `--out_dir`: Directory to save logs and models to (required).
- `--epochs`: Number of epochs to train (default: 20).
- `--num_gpus`: Number of GPUs to use for training (default: 2).
- `--use_val`: Whether to track validation metrics during training (0: false, 1: true). BC4 UNSUPPORTED.
- `--batch_size`: Batch size per process (default: 256).
- `--learning_rate`: Initial learning rate (default: 0.002).
- `--start_momentum`: Initial momentum for SGD (default: 0.9).
- `--end_momentum`: Final momentum for SGD (default: 0.99).
- `--lr_weight_decay`: Weight decay for learning rate (default: 2e-4).
- `--conv1_weight_constraint`: L2 norm constraint for the first convolutional layer
- `--using_windows`: Indicates if the script is running on a Windows machine (0: false, 1: true).
- `--improvements`:Enable improvements and specify type (0: none, 1: blur model).               
- `--early_stopping_patience`: Number of epochs without validation improvement before early stopping (-1: disabled).    


## How to run eval.py
Basic usage:
   ```
   python eval.py --data_dir=<DATA_DIR> --out_dir=<OUT_DIR> --model_path=<MODEL_PATH>
   ```

   Replace <DATA_DIR> with the path to the directory containing the *.pth.tar dataset files as well as the ALLFIXATIONMAPS folder provided in the dataset (if needing evaluation metrics during training) and <OUT_DIR> with the path to the directory where you want the trained model and log to be saved. <MODEL_PATH> is the path of the trained mode. NOTE: IF THE MODEL WAS TRAINED WITH THE IMPROVEMENTS FLAG `--improvements=1`, USE THE FLAG HERE TOO `--improvements=1`

### Full eval.py flags list
- `--data_dir`: Path to the dataset directory containing train_data.pth.tar, val_data.pth.tar (if --use_val=1)(required).
- `--out_dir`: Directory to save logs and models to (required).
- `--model_path`: Path to the model to be evaluated (required).
- `--improvements`: If loading a model that was made using the improvements. 0: none, 1: yes (default=0)
- `--num_saved_images`: Number of comparison images to save when evaluating (-1: all)