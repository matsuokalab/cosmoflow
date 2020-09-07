import os
import numpy as np
import tfrecord
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import horovod.torch as hvd


def load_ds_from_dir(path, batch_size=2):
    # TODO: this is a bit ugly, but I expect to find some torch-y out of the box method later
    tensor_x = []
    tensor_y = []
    # if hvd.rank() == 0:
    #     cnt_files = 4
    # else:
    cnt_files = 4
    for name_file in os.listdir(path)[:cnt_files]:
        path_file = os.path.join(path, name_file)
        reader = tfrecord.reader.tfrecord_loader(data_path=path_file, index_path=None)
        data = next(reader)  # we expect only one record in a file
        x = data["x"].astype(np.float32).reshape(8, 128, 128, 128) / 255 - 0.5
        y = data["y"].astype(np.float32)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tensor_x.append(x)
        tensor_y.append(y)
        # print("in shape:", x.shape)
        # print("in max:", x.max())
        # print("y", y)

    tensor_x = torch.stack(tensor_x)
    # print(f"\n#####worker {hvd.rank()} of {hvd.size()} loading {cnt_files} from {path}\n")
    # print(f"size dataset = {np.prod(tensor_x.shape) * 4 / (1024**2)}M")
    tensor_y = torch.stack(tensor_y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size)


class CFDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self, stage=None):
        print("doing setup")
        # TODO: probably need to scatter indices here by hvd explicitly

    def train_dataloader(self):
        return load_ds_from_dir(os.path.join(self.path, "train"), self.batch_size)

    def val_dataloader(self):
        return load_ds_from_dir(os.path.join(self.path, "validation"), self.batch_size)

    def test_dataloader(self):
        return None
