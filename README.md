# An Improved Multi-Resolution CNN for Predicting Visual Saliency by Lewis Clifton

This repository contains the code for my implementation and improvement of the Mr-CNN saliency prediction model found at:

https://openaccess.thecvf.com/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf

## How to Run

The scripts in this repo assumes it is being run using Blue Crystal 4.

1. Create the conda environment with necessary packages (Pytorch, Pillow, Scipy, Numpy):
   ```
   conda env create -f environment.yml
   ```

2. Edit the SLURM job script train_CNN.sh
   ```
   python train.py --data_dir <DATA_DIR> --out_dir <OUT_DIR>
   ```

   Replace <DATA_DIR> with the path to the directory containing the *.pth.tar dataset files as well as the ALLFIXATIONMAPS folder provided in the dataset (if needing evaluation metrics during training) and <OUT_DIR> with the path to the directory where you want the trained model and log to be saved.

3. Run using SLURM:
   ```
   sbatch train_CNN_job.sh
   ```

## Full flags list

The full list of flags can be given using `python train.py -h` but is also given below:

- `--data_dir`: Path to the dataset directory (required).
- `--out_dir`: Directory to save logs and models to (required).
- `--epochs`: Number of epochs to train (default: 20).
- `--num_gpus`: Number of GPUs to use for training (default: 2).
- `--batch_size`: Batch size per process (default: 256).
- `--learning_rate`: Initial learning rate (default: 0.002).
- `--start_momentum`: Initial momentum for SGD (default: 0.9).
- `--end_momentum`: Final momentum for SGD (default: 0.99).
- `--lr_weight_decay`: Weight decay for learning rate (default: 2e-4).

## Default Hyperparameters

These hyperparameters are default based on the reference paper. These hyperparameters can be changed using command line arguments.

- **Epochs**: 20
- **Batch Size**: 256
- **Learning Rate**: 0.002
- **Start Momentum**: 0.9
- **End Momentum**: 0.99
- **Weight Decay**: 2e-4

To use different hyperparameters, pass them as arguments to the python execution in the SLURM job. For example:
```bash
python train.py --data_dir <DATA_DIR> --out_dir <OUT_DIR> --epochs 15 --batch_size 128
```