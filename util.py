import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
def split_dataset(dataset,shuffle,batch_size,train_ratio,val_ratio):

    train_size = int(len(dataset) *(train_ratio/10.0))
    validate_size = int(len(dataset) * (val_ratio/10.0))
    test_size = len(dataset) - validate_size - train_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                  [train_size, validate_size,
                                                                                   test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return train_loader, validate_loader, test_loader

def load_model(params_small,model_large):
    params_large = model_large.state_dict()
    for k,v in params_small.items():
        layer_name = 'jknet.'+k
        if layer_name in params_large.keys():
            params_large[layer_name] = params_small[k]
        model_large.load_state_dict(params_large)

if __name__ == '__main__':
    split_dataset()