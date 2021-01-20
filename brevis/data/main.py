# import brevis.data as data
from brevis import data as data
from torch.utils.data import DataLoader
import os
import torch
import random
import numpy as np
# import horovod.torch as hvd


def run(config):

    train_dataset = getattr(data, config["dataset"])(config, config["train_csv_file"])
    # train_dataset = BaseDataset(config, config["train_csv_file"])
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        worker_init_fn=worker_init_fn,
        shuffle=config["shuffle"],
        num_workers=16,
    )
    valid_dataset = getattr(data, config["dataset"])(
        config, config["valid_csv_file"], augment=False
    )

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, num_workers=8)
    test_dataset = getattr(data, config["dataset"])(
        config, config["test_csv_file"], augment=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=8
    )  # config["batch_size"])
    return train_dataloader, valid_dataloader, test_dataloader


def worker_init_fn(worker_id=42):
    base_seed = int(torch.randint(2 ** 32, (1,)).item())
    lib_seed = (base_seed + worker_id) % (2 ** 32)
    os.environ["PYTHONHASHSEED"] = str(lib_seed)
    random.seed(lib_seed)
    np.random.seed(lib_seed)
