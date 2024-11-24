import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def save_log(out_dir, date, **kwargs):
    # Save a log
    path = os.path.join(out_dir, 'log.txt')
    with open(path , "w") as f:
        f.write(f"Date/time of creation : {date}\n")
        for k, v in kwargs.items():
            if isinstance(v, list):
                if len(v) == 0: continue
            f.write(f"{k} : {v}\n")

    print(f'Saved log to {path}.')


def setup_gpus(rank, world_size):
    # Setup multiple gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_data_loader(dataset, rank, world_size, batch_size=32):
    # Get the dataloaders
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader