import json
import torch
from torch.utils.data import Dataset, DataLoader


class JSONDataset(Dataset):

    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = torch.tensor(sample['x'])
        y = torch.tensor(sample['y'])
        return x, y


def get_dataloaders(args):

    json_path = 'path/to/your/json/file.json'
    dataset = JSONDataset(json_path)

    # Split the data into training and testing sets
    train_size = int(args.train_path * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(
        dataset, [train_size, valid_size])

    if len(train_data) == 0 or len(valid_data) == 0:
        utils.warnmsg("Warning: length of training or validation data is 0. "
                      "Check --train_perc.")
        if len(valid_data) == 0:
            utils.warnmsg("Warning: length of validation data is 0. "
                          "Assigning all data from the training")
            valid_data = train_data
        else:
            raise ValueError("Length of training data is 0. "
                             "Consider to increase --train_perc.")

    # Create separate data loaders for the training and testing sets
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_data,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True)
    return train_dataloader, valid_dataloader
