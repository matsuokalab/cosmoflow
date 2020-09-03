import numpy as np
import pytorch_lightning as pl
import tfrecord
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import build_model

my_x = [
    np.array([[1.0, 2], [3, 4]]),
    np.array([[5.0, 6], [7, 8]]),
]  # a list of numpy arrays
my_y = [np.array([4.0]), np.array([2.0])]  # another list of numpy arrays (targets)

tensor_x = torch.Tensor(my_x)  # transform to torch tensor
tensor_y = torch.Tensor(my_y)

my_dataloader = DataLoader(my_dataset)  # create your dataloader

path = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small/train/univ_ics_2019-03_a10000668_000.tfrecord"

reader = tfrecord.reader.tfrecord_loader(data_path=path, index_path=None)

for data in reader:
    print(data)
    x = data["x"].astype(np.float32).reshape(8, 128, 128, 128) / 255 - 0.5
    x = np.array([x] * 4)
    y = data["y"].astype(np.float32)
    y = np.array([y] * 4)
    print("in shape:", x.shape)
    print("in max:", x.max())
    print("y", y)
    ####

tensor_x = torch.from_numpy(x)
tensor_y = torch.from_numpy(y)
my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

# write data iterator or reuse off-the shelf something
# either pytorch dataloader or I want to try lightning actually


class LitClassifier(pl.LightningModule):
    def __init__(self):
        self.net = build_model((128, 128, 128, 8), 4, 0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=0.02)


net = build_model((128, 128, 128, 8), 4, 0)
net.training_step = training_step


# extend to work with lightning

result = net(x)
print(result)


# add training loop -
# can also try lightning here
# I actually really loved Chainer's trainer for its extensions architecture
# and it is ported to torch, but not sure that will be good long-term investment :-\

# train ^_^
# add wandb logging
