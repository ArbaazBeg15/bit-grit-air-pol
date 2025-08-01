import random
import numpy as np
import torch
import torch.nn.functional as F    
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def setup_reproducibility(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = config.benchmark
    torch.use_deterministic_algorithms(config.deterministic_algo, warn_only=True)
    torch.set_float32_matmul_precision("high")


def metric_fn(preds, targets):
    mse = F.mse_loss(preds, targets)
    rmse = -(torch.sqrt(mse) / 100)
    return torch.exp(rmse)


def metric_fn_np(preds, targets):
    rmse = root_mean_squared_error(preds, targets)
    rmse = -(rmse / 100)
    return np.exp(rmse)


def load_and_scale_data(path, SEED):
    df = pd.read_csv(path).drop(columns=['id'])
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    #data = df.to_numpy()
    data = scaler.fit_transform(data)
    
    train, eval = train_test_split(data, train_size=0.8, random_state=SEED, shuffle=True)
    train_inputs, train_targets = train[:, :-1], train[:, 6].reshape(-1, 1)
    eval_inputs, eval_targets = eval[:, :-1], eval[:, 6].reshape(-1, 1)
    
    return (
        train_inputs,
        train_targets, 
        eval_inputs,
        eval_targets
    )


def build_loader(
    SEED,
    ds,
    train=True,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    drop_last=True, 
    pin_memory=True,
    persistent_workers=False,
):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    generator = torch.Generator()
    generator.manual_seed(SEED if train else SEED+1)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        #sampler=DistributedSampler(
        #    train_ds,
        #    shuffle=True,
        #    drop_last=True, 
        #    seed=config.seed    
        #)
    )
    
    
class TabularDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]