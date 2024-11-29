import argparse
import os
from improvements import ImprovedMrCNN
from eval import load_model
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help="Path to directory for dataset containing *_data.pth.tar", required=True)
args = parser.parse_args()

model = ImprovedMrCNN().to('cuda')
model = load_model(model, os.path.join(os.getcwd(), args.model_dir))

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'sqrt' in name:
            print(name, round(param.data.item()**2, 4))