import os
import numpy as np
from tfrecord_lite import tf_record_iterator
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import horovod.torch as hvd


def load_ds_from_dir(path, batch_size=2):
    tensor_x = []
    tensor_y = []
    max_files = 256
    # TODO: only read file names on master nodes and then shuffle
    # TODO: do lazy loading if does not fit into memory
    # / check torch built-in tools for this
    files = sorted(os.listdir(path))
    chunks = np.array_split(files, hvd.size())
    local_chunk = chunks[hvd.rank()]
    if hvd.rank() <= 4:
        print(f"!!!!!! r {hvd.rank()} of {hvd.size()} loading {len(local_chunk)} of {len(files)} files")
    # lightning seems to work even if chunks are not equal size!
    for name_file in local_chunk[:max_files]:
        path_file = os.path.join(path, name_file)
        it = tf_record_iterator(path_file)
        data = next(it)
        x = np.frombuffer(data["x"][0], dtype='>u2')
        x = x.reshape(-1, 128, 128, 128)
        x = x.astype(np.float32) / 255 - 0.5
        y = data["y"]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tensor_x.append(x)
        tensor_y.append(y)
        # print("in shape:", x.shape)
        # print("in max:", x.max())
        # print("y", y)

    tensor_x = torch.stack(tensor_x)
    if hvd.rank() <= 4:
        print(f"\n#####worker {hvd.rank()} of {hvd.size()} loaded {tensor_x.shape} from {path}\n")
    # print(f"size dataset = {np.prod(tensor_x.shape) * 4 / (1024**2)}M")
    tensor_y = torch.stack(tensor_y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)


class CFDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self, stage=None):
        # print("doing setup")
        # TODO: probably need to scatter indices here by hvd explicitly
        pass

    def train_dataloader(self):
        return load_ds_from_dir(os.path.join(self.path, "train"), self.batch_size)

    def val_dataloader(self):
        return load_ds_from_dir(os.path.join(self.path, "validation"), self.batch_size)
