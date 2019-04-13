from importlib import import_module
import torch
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, flags):
        self.loader_train = None
        module_train = import_module('data.' + flags.data_train.lower())  # lower:convert uppercase to lowercase
        trainset = getattr(module_train, flags.data_train)(flags)
        self.loader_train = torch.utils.data.DataLoader(
            trainset, batch_size=flags.batch_num,
            drop_last=True,
            shuffle=True,
            num_workers=int(flags.n_threads),
            pin_memory=not flags.cpu
        )


