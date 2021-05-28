import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset):
        train_data = list(zip(*dataset[0]))
        self.data_x, self.s_idx, self.t_idx, self.Ts_vals = train_data[0], train_data[1], train_data[2], train_data[3]
        self.data_y = dataset[1]

    def __getitem__(self, index):
        suvt_embd, s_idx, t_idx, Ts_vals = self.data_x[index], self.s_idx[index], self.t_idx[index], self.Ts_vals[index]
        target = self.data_y[index]
        return suvt_embd, s_idx, t_idx, Ts_vals, target

    def __len__(self):
        return len(self.data_x)


def custom_collate_fn(batch):
    inputs, s_idx, t_idx, Ts_vals, targets = zip(*batch)
    return inputs, s_idx, t_idx, Ts_vals, targets
